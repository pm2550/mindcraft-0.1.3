import { History } from './history.js';
import { Coder } from './coder.js';
import { VisionInterpreter } from './vision/vision_interpreter.js';
import { Prompter } from '../models/prompter.js';
import { initModes } from './modes.js';
import { initBot } from '../utils/mcdata.js';
import { containsCommand, commandExists, executeCommand, truncCommandMessage, isAction, blacklistCommands } from './commands/index.js';
import { ActionManager } from './action_manager.js';
import { NPCContoller } from './npc/controller.js';
import { MemoryBank } from './memory_bank.js';
import { SelfPrompter } from './self_prompter.js';
import convoManager from './conversation.js';
import { handleTranslation, handleEnglishTranslation } from '../utils/translator.js';
import { addBrowserViewer } from './vision/browser_viewer.js';
import { serverProxy, sendOutputToServer } from './mindserver_proxy.js';
import { logEpisodeStep } from './episode_logger.js';
import { PolicyMiner } from './policy_miner.js';
import { SurvivalPolicy } from './survival_policy.js';
import { MicroController } from './micro_controller.js';
import { RTScheduler } from './rt_scheduler.js';
import { ControlArbiter } from './control_arbiter.js';
import { SkillDemoLogger } from './skill_demo_logger.js';
import { SkillRegistry } from './skill_registry.js';
import { registerBuiltinSurvivalSkills } from './builtin_survival_skills.js';
import settings from './settings.js';
import { Task } from './tasks/tasks.js';
import { speak } from './speak.js';

export class Agent {
    async start(load_mem=false, init_message=null, count_id=0) {
        this.last_sender = null;
        this.count_id = count_id;
        this.warmup_until = 0;
        this.warmup_active = false;
        this._plannerCommandLoop = { cmd: '', count: 0, lastAt: 0 };

        // Initialize components with more detailed error handling
        this.actions = new ActionManager(this);
        this.prompter = new Prompter(this, settings.profile);
        this.name = this.prompter.getName();
        console.log(`Initializing agent ${this.name}...`);
        this.history = new History(this);
        this.coder = new Coder(this);
        this.npc = new NPCContoller(this);
        this.memory_bank = new MemoryBank();
        this.self_prompter = new SelfPrompter(this);
        this.policyMiner = new PolicyMiner(`./bots/${this.name}`);
        this.policyMiner.mine();
        setInterval(() => {
            try { this.policyMiner.mine(); } catch (e) { console.error('Policy mining error:', e.message); }
        }, 300000);

        // === Real-time pipeline ===
        this.scheduler = new RTScheduler(this);
        this.survivalPolicy = new SurvivalPolicy(this, `./bots/${this.name}`);
        this.microCtrl = new MicroController(this, `./bots/${this.name}`);
        this.control = new ControlArbiter(this);
        this.skillDemoLogger = new SkillDemoLogger(this.name);
        this.skillRegistry = registerBuiltinSurvivalSkills(new SkillRegistry(this, this.skillDemoLogger));
        this._survivalFastCooldownUntil = 0;
        // Fast reactive check every 500ms — only interrupts when there is an actionable reflex
        // Highest-priority owner: emergency survival can preempt planner.
        this._survivalTimer = setInterval(async () => {
            if (!this.bot?.entity) return;
            if (this._survivalBusy) return;
            if (this.warmup_active || Date.now() < (this.warmup_until || 0)) return;
            if (Date.now() < this._survivalFastCooldownUntil) return;
            this.scheduler?.checkOverrideTimeout?.();
            // Allow reflex if: not in survival mode, OR critical HP during fight
            const activeMode = this.scheduler?.currentMode;
            const state0 = this.survivalPolicy?._quickCheck();
            if (activeMode === 'fight') {
                // Don't interrupt fight unless HP is critical — taking damage while fighting is normal
                if ((state0?.health || 20) > 6) return;
            } else if ((activeMode === 'flee' || activeMode === 'hide') && !state0?.recentDamage) {
                return; // already fleeing/hiding and not taking new damage
            }
            try {
                const state = this.survivalPolicy._quickCheck();
                if (!state) return;
                const hasThreat = state.mobCount > 0 && state.closestMobDist < 16;
                const safeToEatNow =
                    state.totalFood > 0 &&
                    state.bestFood &&
                    state.health <= 16 &&
                    (state.hunger ?? 20) <= 17 &&
                    (!hasThreat || state.closestMobDist > 10) &&
                    !state.recentDamage;

                let fastAction = null;
                if (hasThreat && (state.closestMobDist < 12 || state.recentDamage)) {
                    // Use combat assessment instead of blindly fleeing
                    const combat = this.survivalPolicy.assessCombat(state);
                    if (combat.threatLevel === 'none' || combat.threatLevel === 'trivial') {
                        // Trivial threat — fight if close, ignore if far
                        fastAction = state.closestMobDist < 8 ? 'fight' : null;
                    } else if (combat.threatLevel === 'manageable') {
                        // Can win — fight
                        fastAction = 'fight';
                    } else {
                        // Dangerous or lethal — flee
                        fastAction = 'flee';
                    }
                    // Ranged override: NEVER fight skeleton/stray/pillager.
                    // Bot can't close distance to ranged mobs (152 fight frames, 0 close, 0 attacks).
                    // Flee from ranged is the only viable option.
                    if (state.closestMobIsRanged) fastAction = 'flee';
                    // Critical HP override: hide (dig shelter) instead of flee.
                    // hp<=6 was too late — bot died at hp=0.5 while digging. Need hp<=10
                    // to give enough time for digDown(3) to complete (1.6 seconds).
                    if (state.health <= 10) fastAction = 'hide';
                    // Creeper override: always flee (any distance)
                    if (state.closestMobIsExplosive) fastAction = 'flee';
                    // Enderman override: always flee. 40hp + teleport + 7dps = unfightable without gear.
                    // Looking at enderman triggers aggro — fleeing (not looking) is the only safe option.
                    if (state.closestMobType === 'enderman') fastAction = 'flee';
                } else if (safeToEatNow) {
                    fastAction = 'eat';
                }
                if (!fastAction) return;

                this._survivalBusy = true;
                console.log(`[Reflex] hp=${state.health} mob=${state.closestMobType}@${state.closestMobDist} → ${fastAction}`);
                // Log reflex event to survival episodes with source tag
                if (this.survivalPolicy) {
                    this.survivalPolicy.logReflex(fastAction, state);
                }
                const continuousMode = fastAction === 'flee' || fastAction === 'fight' || fastAction === 'hide';
                const enteringContinuousMode = continuousMode && activeMode !== fastAction;
                if (this.scheduler) {
                    this.scheduler.setMode(fastAction, 'reflex');
                    // All continuous combat modes (flee/fight/hide) need 10s TTL.
                    // 3s was too short for flee — creeper TTL expired before bot escaped blast range.
                    // Flee is extended by postFleeUntil but 3+4=7s still not enough for skeleton kiting.
                    const reflexTtl = continuousMode ? 10000 : 2500;
                    this.scheduler.declareOverride(
                        fastAction,
                        `reflex hp=${state.health} mob=${state.closestMobType}@${state.closestMobDist}`,
                        reflexTtl,
                        'return_anchor',
                        'reflex',
                    );
                    // Reflex flee also sets postFleeUntil so override timeout
                    // won't drop to continue while threat is still close
                    if (fastAction === 'flee') {
                        this.scheduler.postFleeUntil = Date.now() + 10000;
                    }
                }
                if (continuousMode) {
                    this.microCtrl?.kickObservation?.('reflex');
                }
                if (enteringContinuousMode) {
                    try { await this.bot?._bridge?.sendCommand('stop'); } catch (e) {}
                }
                // Only eat is discrete (needs server response). Flee is handled by micro.
                if (fastAction === 'eat' && state.totalFood > 0 && state.bestFood) {
                    try {
                        const skills = await import('./library/skills.js');
                        await Promise.race([
                            skills.consume(this.bot, state.bestFood),
                            new Promise((_, rej) => setTimeout(() => rej(), 2500)),
                        ]);
                        console.log(`[Reflex] Ate ${state.bestFood}`);
                    } catch (e) {}
                }
                // flee/fight: no direct execution. Scheduler mode is set, micro handles movement.
                this._survivalFastCooldownUntil = Date.now() + 800;
                this._survivalBusy = false;
            } catch (e) {
                this._survivalBusy = false;
            }
        }, 500);

        // Start continuous micro controller (publishes proposals to scheduler)
        // Delay until bot is fully connected
        setTimeout(() => {
            if (this.bot?._bridge) {
                this.microCtrl.start();
            }
        }, 5000);

        // Scheduler tick: every 50ms resolve best proposal and send to game
        this._schedulerTick = setInterval(() => {
            if (!this.bot?._bridge || !this.scheduler) return;
            if (this.warmup_active || Date.now() < (this.warmup_until || 0)) return;
            this.scheduler.checkOverrideTimeout?.();
            const action = this.scheduler.resolve();
            if (action) {
                try { this.bot._bridge.sendCommand('rl_action', action).catch(() => {}); } catch(e) {}
            }
        }, 50);

        convoManager.initAgent(this);
        await this.prompter.initExamples();

        // load mem first before doing task
        let save_data = null;
        if (load_mem) {
            save_data = this.history.load();
        }
        let taskStart = null;
        if (save_data) {
            taskStart = save_data.taskStart;
        } else {
            taskStart = Date.now();
        }
        this.task = new Task(this, settings.task, taskStart);
        this.blocked_actions = settings.blocked_actions.concat(this.task.blocked_actions || []);
        blacklistCommands(this.blocked_actions);

        console.log(this.name, 'logging into minecraft...');
        this.bot = await initBot(this.name);

        initModes(this);

        this.bot.on('login', () => {
            console.log(this.name, 'logged in!');
            serverProxy.login();
            
            // Set skin for profile, requires Fabric Tailor. (https://modrinth.com/mod/fabrictailor)
            if (this.prompter.profile.skin)
                this.bot.chat(`/skin set URL ${this.prompter.profile.skin.model} ${this.prompter.profile.skin.path}`);
            else
                this.bot.chat(`/skin clear`);
        });

        const spawnTimeoutMs = settings.spawn_timeout_ms ?? 30000;
        const spawnTimeout = spawnTimeoutMs > 0 ? setTimeout(() => {
            console.error(`Bot has not spawned after ${Math.round(spawnTimeoutMs / 1000)} seconds. Exiting.`);
            process.exit(1);
        }, spawnTimeoutMs) : null;
        this.bot.once('spawn', async () => {
            try {
                if (spawnTimeout) {
                    clearTimeout(spawnTimeout);
                }
                addBrowserViewer(this.bot, count_id);
                console.log('Initializing vision intepreter...');
                this.vision_interpreter = new VisionInterpreter(this, settings.allow_vision);

                // wait for a bit so stats are not undefined
                await new Promise((resolve) => setTimeout(resolve, 1000));
                
                console.log(`${this.name} spawned.`);
                this.clearBotLogs();

                this._enterWarmup(settings.spawn_warmup_ms ?? 0, 'initial_spawn');
              
                this._setupEventHandlers(save_data, init_message);
                this.startEvents();
              
                if (!load_mem) {
                    if (settings.task) {
                        this.task.initBotTask();
                        this.task.setAgentGoal();
                    }
                } else {
                    // set the goal without initializing the rest of the task
                    if (settings.task) {
                        this.task.setAgentGoal();
                    }
                }

                await new Promise((resolve) => setTimeout(resolve, 10000));
                this.checkAllPlayersPresent();

            } catch (error) {
                console.error('Error in spawn event:', error);
                process.exit(1);
            }
        });
    }

    async _setupEventHandlers(save_data, init_message) {
        const ignore_messages = [
            "Set own game mode to",
            "Set the time to",
            "Set the difficulty to",
            "Teleported ",
            "Set the weather to",
            "Gamerule "
        ];
        
        const respondFunc = async (username, message) => {
            if (message === "") return;
            if (username === this.name) return;
            if (settings.only_chat_with.length > 0 && !settings.only_chat_with.includes(username)) return;
            try {
                if (ignore_messages.some((m) => message.startsWith(m))) return;

                this.shut_up = false;

                console.log(this.name, 'received message from', username, ':', message);

                if (convoManager.isOtherAgent(username)) {
                    console.warn('received whisper from other bot??')
                }
                else {
                    let translation = await handleEnglishTranslation(message);
                    this.handleMessage(username, translation);
                }
            } catch (error) {
                console.error('Error handling message:', error);
            }
        }

		this.respondFunc = respondFunc;

        this.bot.on('whisper', respondFunc);
        
        this.bot.on('chat', (username, message) => {
            if (serverProxy.getNumOtherAgents() > 0) return;
            // only respond to open chat messages when there are no other agents
            respondFunc(username, message);
        });

        // Set up auto-eat — enable Java-side auto-eat via bridge command
        this.bot.autoEat.options = {
            priority: 'foodPoints',
            startAt: 14,
            bannedFood: ["rotten_flesh", "spider_eye", "poisonous_potato", "pufferfish", "chicken"]
        };
        // Enable Java-side auto-eat (food<10 triggers eat from inventory)
        if (this.bot._bridge) {
            this.bot._bridge.sendCommand('autoEat', { enabled: true }).catch(() => {});
        }
        // TODO: auto-armor needs Java-side implementation (CommandHandler doesn't have autoArmor yet)

        const warmupDelay = this.warmup_until ? Math.max(0, this.warmup_until - Date.now()) : 0;

        if (save_data?.self_prompt) {
            if (init_message) {
                this.history.add('system', init_message);
            }
            await this.self_prompter.handleLoad(save_data.self_prompt, save_data.self_prompting_state);
        }
        if (save_data?.last_sender) {
            this.last_sender = save_data.last_sender;
            if (convoManager.otherAgentInGame(this.last_sender)) {
                const msg_package = {
                    message: `You have restarted and this message is auto-generated. Continue the conversation with me.`,
                    start: true
                };
                convoManager.receiveFromBot(this.last_sender, msg_package);
            }
        }
        else if (init_message) {
            const sendInit = async () => {
                // !goal(...) is executed directly to guarantee activation
                if (init_message.startsWith('!goal')) {
                    console.log('Executing startup goal directly:', init_message);
                    await executeCommand(this, init_message);
                } else {
                    await this.handleMessage('system', init_message, 2);
                }
            };
            if (warmupDelay > 0) {
                setTimeout(sendInit, warmupDelay);
            } else {
                await sendInit();
            }
        }
        else {
            const sayHello = () => this.openChat("Hello world! I am "+this.name);
            if (warmupDelay > 0) {
                setTimeout(sayHello, warmupDelay);
            } else {
                sayHello();
            }
        }
    }

    checkAllPlayersPresent() {
        if (!this.task || !this.task.agent_names) {
          return;
        }

        const missingPlayers = this.task.agent_names.filter(name => !this.bot.players[name]);
        if (missingPlayers.length > 0) {
            console.log(`Missing players/bots: ${missingPlayers.join(', ')}`);
            this.cleanKill('Not all required players/bots are present in the world. Exiting.', 4);
        }
    }

    requestInterrupt() {
        this.bot.interrupt_code = true;
        this.bot.stopDigging();
        this.bot.collectBlock.cancelTask();
        this.bot.pathfinder.stop();
        this.bot.pvp.stop();
    }

    clearBotLogs() {
        this.bot.output = '';
        this.bot.interrupt_code = false;
    }

    shutUp() {
        this.shut_up = true;
        if (this.self_prompter.isActive()) {
            this.self_prompter.stop(false);
        }
        convoManager.endAllConversations();
    }

    async handleMessage(source, message, max_responses=null) {
        await this.checkTaskDone();
        if (!source || !message) {
            console.warn('Received empty message from', source);
            return false;
        }
        const raw_message = message;

        let used_command = false;
        if (max_responses === null) {
            max_responses = settings.max_commands === -1 ? Infinity : settings.max_commands;
        }
        if (max_responses === -1) {
            max_responses = Infinity;
        }

        const self_prompt = source === 'system' || source === this.name;
        const from_other_bot = convoManager.isOtherAgent(source);
        const from_user = !self_prompt && !from_other_bot;
        let sent_user_reply = false;

        if (from_user) { // from user, check for forced commands
            const user_command_name = containsCommand(message);
            if (user_command_name) {
                if (!commandExists(user_command_name)) {
                    if (!sent_user_reply) {
                        this.routeResponse(source, `Command '${user_command_name}' does not exist.`);
                        sent_user_reply = true;
                    }
                    return false;
                }
                if (!sent_user_reply) {
                    this.routeResponse(source, `*${source} used ${user_command_name.substring(1)}*`);
                    sent_user_reply = true;
                }
                if (user_command_name === '!newAction') {
                    // all user-initiated commands are ignored by the bot except for this one
                    // add the preceding message to the history to give context for newAction
                    this.history.add(source, message);
                }
                let execute_res = await executeCommand(this, message);
                if (execute_res && !sent_user_reply) {
                    this.routeResponse(source, execute_res);
                    sent_user_reply = true;
                }
                return true;
            }
        }

        if (from_other_bot)
            this.last_sender = source;

        // Now translate the message
        message = await handleEnglishTranslation(message);
        console.log('received message from', source, ':', message);
        if (from_user) {
            this.last_user_message = message;
            this.last_user_message_raw = raw_message;
        }
        this.last_command_context = {
            source,
            from_user,
            self_prompt,
            raw_message,
            message
        };

        const checkInterrupt = () => this.self_prompter.shouldInterrupt(self_prompt) || this.shut_up || convoManager.responseScheduledFor(source);
        
        let behavior_log = this.bot.modes.flushBehaviorLog().trim();
        if (behavior_log.length > 0) {
            const MAX_LOG = 500;
            if (behavior_log.length > MAX_LOG) {
                behavior_log = '...' + behavior_log.substring(behavior_log.length - MAX_LOG);
            }
            behavior_log = 'Recent behaviors log: \n' + behavior_log;
            await this.history.add('system', behavior_log);
        }

        // Handle other user messages
        await this.history.add(source, message);
        this.history.save();

        if (!self_prompt && this.self_prompter.isActive()) // message is from user during self-prompting
            max_responses = 1; // force only respond to this message, then let self-prompting take over
        for (let i=0; i<max_responses; i++) {
            if (checkInterrupt()) break;
            let history = this.history.getHistory();
            let res = await this.prompter.promptConvo(history);

            console.log(`${this.name} full response to ${source}: ""${res}""`);

            if (res.trim().length === 0) {
                console.warn('no response')
                break; // empty response ends loop
            }

            let command_name = containsCommand(res);

            if (command_name) { // contains query or command
                res = truncCommandMessage(res); // everything after the command is ignored
                this.history.add(this.name, res);

                // Generic command loop detection: any self-prompt command repeated 3+ times
                if (self_prompt) {
                    const now = Date.now();
                    // Extract just the command + args from the response, not the reasoning text
                    const cmdMatch = res.match(new RegExp(command_name.replace('!', '!') + '\\s*\\([^)]*\\)|' + command_name.replace('!', '!') + '\\s*"[^"]*"'));
                    const cmdKey = cmdMatch ? cmdMatch[0].slice(0, 80) : command_name;
                    const prev = this._plannerCommandLoop || { cmd: '', count: 0, lastAt: 0 };
                    if (prev.cmd === cmdKey && (now - prev.lastAt) < 300000) {
                        this._plannerCommandLoop = { cmd: cmdKey, count: prev.count + 1, lastAt: now };
                    } else {
                        this._plannerCommandLoop = { cmd: cmdKey, count: 1, lastAt: now };
                    }
                    if (this._plannerCommandLoop.count >= 3) {
                        this.history.add('system',
                            'Loop detected: same command repeated 3+ times. This approach is NOT working. ' +
                            'Try a COMPLETELY different strategy: move to a new area with !baritone("explore"), ' +
                            'try different block/item types, or check !nearbyBlocks / !inventory for alternatives.');
                        console.warn(`[Planner] Loop suppressed: ${cmdKey} (${this._plannerCommandLoop.count}x)`);
                        continue;
                    }
                }
                
                if (!commandExists(command_name)) {
                    this.history.add('system', `Command ${command_name} does not exist.`);
                    console.warn('Agent hallucinated command:', command_name)
                    continue;
                }

                const rtMode = this.scheduler?.currentMode;
                const plannerBlockedByWarmup =
                    self_prompt &&
                    (this.warmup_active || Date.now() < (this.warmup_until || 0));
                const plannerBlockedByRealtime =
                    self_prompt &&
                    (this.scheduler?.plannerInhibited || rtMode === 'flee' || rtMode === 'fight' || rtMode === 'hide');
                if (plannerBlockedByWarmup || plannerBlockedByRealtime) {
                    const evt = this.scheduler?.getPlannerEvent?.();
                    const why = plannerBlockedByWarmup
                        ? 'warmup'
                        : (evt ? `${evt.mode}: ${evt.reason}` : `mode=${rtMode}`);
                    this.history.add('system', `Planner command suppressed by runtime control (${why}).`);
                    console.log(`[Planner] Suppressed self-prompt command during runtime control (${why})`);
                    break;
                }

                if (checkInterrupt()) break;
                this.self_prompter.handleUserPromptedCmd(self_prompt, isAction(command_name));

                const should_send = !self_prompt && (!from_user || !sent_user_reply);
                if (should_send) {
                    if (settings.show_command_syntax === "full") {
                        this.routeResponse(source, res);
                    }
                    else if (settings.show_command_syntax === "shortened") {
                        // show only "used !commandname"
                        let pre_message = res.substring(0, res.indexOf(command_name)).trim();
                        let chat_message = `*used ${command_name.substring(1)}*`;
                        if (pre_message.length > 0)
                            chat_message = `${pre_message}  ${chat_message}`;
                        this.routeResponse(source, chat_message);
                    }
                    else {
                        // no command at all
                        let pre_message = res.substring(0, res.indexOf(command_name)).trim();
                        if (pre_message.trim().length > 0)
                            this.routeResponse(source, pre_message);
                    }
                    if (from_user) {
                        sent_user_reply = true;
                    }
                }

                const preState = logEpisodeStep.captureState(this);
                let execute_res = await executeCommand(this, res);
                const postState = logEpisodeStep.captureState(this);
                logEpisodeStep(this, command_name, res, execute_res, preState, postState);

                console.log('Agent executed:', command_name, 'and got:', execute_res);
                used_command = true;

                if (execute_res)
                    this.history.add('system', execute_res);
                else
                    break;

                // Self-prompting is a planner step: execute exactly one command, then return
                // so the outer self-prompt loop can re-run safety and idle checks.
                if (self_prompt) {
                    this.history.save();
                    break;
                }
            }
            else { // conversation response
                this.history.add(this.name, res);
                if (!self_prompt && (!from_user || !sent_user_reply)) {
                    this.routeResponse(source, res);
                    if (from_user) {
                        sent_user_reply = true;
                    }
                }
                break;
            }
            
            this.history.save();
        }

        return used_command;
    }

    async routeResponse(to_player, message) {
        if (this.shut_up) return;
        let self_prompt = to_player === 'system' || to_player === this.name;
        if (self_prompt && this.last_sender) {
            // this is for when the agent is prompted by system while still in conversation
            // so it can respond to events like death but be routed back to the last sender
            to_player = this.last_sender;
        }

        if (convoManager.isOtherAgent(to_player) && convoManager.inConversation(to_player)) {
            // if we're in an ongoing conversation with the other bot, send the response to it
            convoManager.sendToBot(to_player, message);
        }
        else {
            // otherwise, use open chat
            this.openChat(message);
            // note that to_player could be another bot, but if we get here the conversation has ended
        }
    }

    async openChat(message) {
        let to_translate = message;
        let remaining = '';
        let command_name = containsCommand(message);
        let translate_up_to = command_name ? message.indexOf(command_name) : -1;
        if (translate_up_to != -1) { // don't translate the command
            to_translate = to_translate.substring(0, translate_up_to);
            remaining = message.substring(translate_up_to);
        }
        message = (await handleTranslation(to_translate)).trim() + " " + remaining;
        // newlines are interpreted as separate chats, which triggers spam filters. replace them with spaces
        message = message.replaceAll('\n', ' ');

        if (settings.only_chat_with.length > 0) {
            for (let username of settings.only_chat_with) {
                this.bot.whisper(username, message);
            }
        }
        else {
            if (settings.speak) {
                speak(to_translate, this.prompter.profile.speak_model);
            }
            if (settings.chat_ingame) {this.bot.chat(message);}
            sendOutputToServer(this.name, message);
        }
    }

    async _resetRealtimeControl(reason = 'reset') {
        try { this.scheduler?.reset(reason); } catch (err) {}
        try { this.microCtrl?.resetEpisode(reason); } catch (err) {}
        this._survivalBusy = false;
        this._survivalFastCooldownUntil = 0;
        // Clear Baritone busy flag so isIdle() returns true after death/spawn
        if (this.bot) {
            this.bot._baritoneActive = false;
            this.bot._baritoneBusyUntil = 0;
        }
        try {
            await this.bot?._bridge?.sendCommand('interrupt', {}, 5000);
        } catch (err) {}
        try {
            await this.bot?._bridge?.sendCommand('rl_action', {
                forward: false, back: false, left: false, right: false,
                jump: false, sprint: false, sneak: false,
                attack: false, use: false,
                dyaw: 0, dpitch: 0,
            }, 5000);
        } catch (err) {}
    }

    _enterWarmup(ms, reason = 'warmup') {
        const warmupMs = Math.max(0, ms || 0);
        if (this._warmupTimer) {
            clearTimeout(this._warmupTimer);
            this._warmupTimer = null;
        }
        if (warmupMs <= 0) {
            this.warmup_until = 0;
            this.warmup_active = false;
            return;
        }
        this.warmup_until = Date.now() + warmupMs;
        this.warmup_active = true;
        console.log(`[Agent] warmup ${warmupMs}ms (${reason})`);
        if (this.bot?.modes?.pauseAll) {
            this.bot.modes.pauseAll();
        }
        this._warmupTimer = setTimeout(() => {
            this.warmup_active = false;
            this.warmup_until = 0;
            if (this.bot?.modes?.unPauseAll) {
                this.bot.modes.unPauseAll();
            }
            this._warmupTimer = null;
            console.log(`[Agent] warmup ended (${reason})`);
            if (this.self_prompter?.isActive?.() && !this.self_prompter.loop_active) {
                console.log(`[Agent] resuming planner after warmup (${reason})`);
                this.self_prompter.startLoop();
            }
        }, warmupMs);
    }

    _stringifyReason(reason) {
        if (reason === undefined || reason === null) return 'unknown';
        if (typeof reason === 'string') return reason;
        try {
            return JSON.stringify(reason);
        } catch (err) {
            return String(reason);
        }
    }

    async logDisconnectEvent(kind, reason) {
        const reasonText = this._stringifyReason(reason);
        const pos = this.bot?.entity?.position;
        const ctx = {
            kind,
            reason: reasonText,
            action: this.actions?.currentActionLabel || '',
            position: pos ? { x: pos.x, y: pos.y, z: pos.z } : null,
            health: this.bot?.health,
            food: this.bot?.food,
            dimension: this.bot?.game?.dimension,
            timestamp: new Date().toISOString()
        };
        try {
            await this.history.add('system', `Disconnect event: ${ctx.kind} (${ctx.reason}) | action=${ctx.action || 'none'} | pos=${ctx.position ? `${ctx.position.x.toFixed(2)}, ${ctx.position.y.toFixed(2)}, ${ctx.position.z.toFixed(2)}` : 'unknown'} | health=${ctx.health ?? 'n/a'} | food=${ctx.food ?? 'n/a'} | dim=${ctx.dimension ?? 'unknown'}`);
            await this.history.save();
        } catch (err) {
            console.error('Failed to write disconnect event to history:', err);
        }
    }

    startEvents() {
        // Custom events
        this.bot.on('time', () => {
            if (this.bot.time.timeOfDay == 0)
            this.bot.emit('sunrise');
            else if (this.bot.time.timeOfDay == 6000)
            this.bot.emit('noon');
            else if (this.bot.time.timeOfDay == 12000)
            this.bot.emit('sunset');
            else if (this.bot.time.timeOfDay == 18000)
            this.bot.emit('midnight');
        });

        let prev_health = this.bot.health;
        this.bot.lastDamageTime = 0;
        this.bot.lastDamageTaken = 0;
        this.bot.on('health', () => {
            if (this.bot.health < prev_health) {
                this.bot.lastDamageTime = Date.now();
                this.bot.lastDamageTaken = prev_health - this.bot.health;
            }
            prev_health = this.bot.health;
        });
        // Logging callbacks
        this.bot.on('error' , (err) => {
            console.error('Error event!', err);
        });
        this.bot.on('end', async (reason) => {
            console.warn('Bot disconnected!', reason);
            await this.logDisconnectEvent('end', reason);
            // Exit with code 0 so agent_process.js auto-restarts (code > 1 = permanent exit)
            console.log('Exiting for auto-restart...');
            process.exit(0);
        });
        this.bot.on('death', () => {
            this.actions.cancelResume();
            this.actions.stop();
            this._resetRealtimeControl('death');
            // Write terminal episode for RL training
            if (this.survivalPolicy) {
                this.survivalPolicy.logDeath();
            }
            // Notify micro controller for dynamic explore rate adjustment
            if (this.microCtrl) {
                this.microCtrl.onDeath();
            }
        });
        this.bot.on('spawn', () => {
            this._resetRealtimeControl('spawn');
            const respawnWarmupMs = settings.respawn_warmup_ms ?? settings.spawn_warmup_ms ?? 3000;
            this._enterWarmup(respawnWarmupMs, 'respawn');
        });
        this.bot.on('kicked', async (reason) => {
            console.warn('Bot kicked!', reason);
            await this.logDisconnectEvent('kicked', reason);
            this.cleanKill('Bot kicked! Killing agent process.');
        });
        this.bot.on('messagestr', async (message, _, jsonMsg) => {
            if (jsonMsg.translate && jsonMsg.translate.startsWith('death') && message.startsWith(this.name)) {
                console.log('Agent died: ', message);
                if (this.survivalPolicy) {
                    this.survivalPolicy.logDeath();
                }
                let death_pos = this.bot.entity.position;
                this.memory_bank.rememberPlace('last_death_position', death_pos.x, death_pos.y, death_pos.z);
                let death_pos_text = null;
                if (death_pos) {
                    death_pos_text = `x: ${death_pos.x.toFixed(2)}, y: ${death_pos.y.toFixed(2)}, z: ${death_pos.x.toFixed(2)}`;
                }
                let dimention = this.bot.game.dimension;
                this.handleMessage('system', `You died at position ${death_pos_text || "unknown"} in the ${dimention} dimension with the final message: '${message}'. Your place of death is saved as 'last_death_position' if you want to return. Previous actions were stopped and you have respawned.`);
            }
        });
        this.bot.on('idle', () => {
            this.bot.clearControlStates();
            this.bot.pathfinder.stop(); // clear any lingering pathfinder
            if (this.warmup_active) {
                return;
            }
            this.bot.modes.unPauseAll();
            setTimeout(() => {
                if (!this.warmup_active && this.isIdle()) {
                    this.actions.resumeAction();
                }
            }, 1000);
        });

        // Init NPC controller
        this.npc.init();

        // This update loop ensures that each update() is called one at a time, even if it takes longer than the interval
        const INTERVAL = 300;
        let last = Date.now();
        setTimeout(async () => {
            while (true) {
                let start = Date.now();
                await this.update(start - last);
                let remaining = INTERVAL - (Date.now() - start);
                if (remaining > 0) {
                    await new Promise((resolve) => setTimeout(resolve, remaining));
                }
                last = start;
            }
        }, INTERVAL);

        this.bot.emit('idle');
    }

    async update(delta) {
        await this.bot.modes.update();
        this.self_prompter.update(delta);
        await this.checkTaskDone();
    }

    isIdle() {
        if (this.actions.executing) return false;
        if (this.bot?.isBaritoneBusy && this.bot.isBaritoneBusy()) return false;
        return true;
    }
    

    cleanKill(msg='Killing agent process...', code=1) {
        this.history.add('system', msg);
        this.bot.chat(code > 1 ? 'Restarting.': 'Exiting.');
        this.history.save();
        process.exit(code);
    }
    async checkTaskDone() {
        if (this.task.data) {
            let res = this.task.isDone();
            if (res) {
                await this.history.add('system', `Task ended with score : ${res.score}`);
                await this.history.save();
                // await new Promise(resolve => setTimeout(resolve, 3000)); // Wait 3 second for save to complete
                console.log('Task finished:', res.message);
                this.killAll();
            }
        }
    }

    killAll() {
        serverProxy.shutdown();
    }
}
