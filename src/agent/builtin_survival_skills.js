import { SkillContract, getLatestObs } from './skill_contract.js';
import { PolicyClient } from './policy_client.js';
import fs from 'fs';
import path from 'path';

// Shared skill demo writer — writes obs+action to combat_demos.jsonl during skill execution.
// This solves the freezeMicro problem: skills know what actions they're performing,
// so they can record directly instead of relying on micro controller.
let _skillDemoStream = null;
let _skillDemoCount = 0;
function writeSkillDemo(agent, action, mode) {
    const obs = getLatestObs(agent);
    if (!obs) return;
    if (!_skillDemoStream) {
        const demoPath = path.join('./bots', agent.name || 'GaoBot1', 'combat_demos.jsonl');
        _skillDemoStream = fs.createWriteStream(demoPath, { flags: 'a' });
    }
    const constraints = { _mode: mode };
    const policyObs = PolicyClient.formatObs(obs, constraints, mode);
    if (!policyObs) return;
    const demo = {
        obs: policyObs,
        action: {
            move_fwd: action.forward ? 1.0 : (action.back ? -1.0 : 0.0),
            move_strafe: action.right ? 1.0 : (action.left ? -1.0 : 0.0),
            yaw: action.dyaw || 0,
            pitch: action.dpitch || 0,
            jump: action.jump ? 1.0 : 0.0,
            sprint: action.sprint ? 1.0 : 0.0,
            attack: action.attack ? 1.0 : 0.0,
            use: action.use ? 1.0 : 0.0,
            sneak: action.sneak ? 1.0 : 0.0,
        },
        mode,
        tick: obs.tick || 0,
    };
    _skillDemoStream.write(JSON.stringify(demo) + '\n');
    _skillDemoCount++;
    if (_skillDemoCount % 100 === 0) {
        console.log(`[SkillDemo] ${_skillDemoCount} skill frames recorded`);
    }
}

function findSealBlock(bot) {
    return bot?.inventory?.items?.().find(i =>
        i.name.includes('dirt') ||
        i.name.includes('cobblestone') ||
        i.name.includes('planks') ||
        i.name.includes('stone')
    ) || null;
}

class DigShelterSkill extends SkillContract {
    constructor() {
        super({
            name: 'dig_shelter',
            behaviorId: 1,
            owner: 'script',
            priority: 2,
            timeoutMs: 5000,
            cooldownMs: 4000,
        });
    }

    checkPrecondition(ctx) {
        const obs = getLatestObs(ctx.agent);
        return !!ctx.bot?.entity && !obs?.inWater;
    }

    checkSuccess(ctx, startObs) {
        const endObs = getLatestObs(ctx.agent);
        const startY = ctx.startPosition?.y ?? ctx.bot?.entity?.position?.y ?? 0;
        const endY = ctx.bot?.entity?.position?.y ?? startY;
        const yDelta = endY - startY;
        if (!startObs && !endObs) return false;
        return yDelta <= -2 || (endObs.threats || []).length === 0;
    }

    async execute(ctx) {
        const skills = await import('./library/skills.js');
        const now = Date.now();
        ctx.setTarget(null);
        ctx.setActionMask({
            forward: false, back: false, left: false, right: false,
            jump: false, sprint: false, attack: false, use: true, sneak: true, hotbar: true,
        });
        ctx.scheduler?.setConstraints?.({
            freezeMicro: true,
            freezeMicroUntil: now + 3500,
        });
        ctx.scheduler?.clearProposal?.('micro');
        try {
            await ctx.bot?._bridge?.sendCommand?.('rl_action', {
                forward: false, back: false, left: false, right: false,
                jump: false, sprint: false, attack: false, use: false, sneak: false,
                dyaw: 0, dpitch: 0,
            });
        } catch (e) {}

        ctx.setPhase(1, 0.1, 'dig_down');
        // Record dig action demos: looking down + breaking block below
        const digInterval = setInterval(() => {
            writeSkillDemo(ctx.agent, {
                forward: false, back: false, left: false, right: false,
                jump: false, sprint: false, attack: true, use: false, sneak: true,
                dyaw: 0, dpitch: 90, // looking straight down
            }, 'hide');
        }, 200); // record every 200ms during dig
        await skills.digDown(ctx.bot, 3);
        clearInterval(digInterval);

        const pos = ctx.bot?.entity?.position;
        const sealBlock = findSealBlock(ctx.bot);
        if (sealBlock && pos) {
            ctx.setPhase(2, 0.8, 'seal_top');
            // Record seal action demos: looking up + placing block
            for (let i = 0; i < 5; i++) {
                writeSkillDemo(ctx.agent, {
                    forward: false, back: false, left: false, right: false,
                    jump: false, sprint: false, attack: false, use: true, sneak: true,
                    dyaw: 0, dpitch: -90, // looking up to place seal
                }, 'hide');
            }
            await skills.placeBlock(
                ctx.bot,
                sealBlock.name,
                Math.floor(pos.x),
                Math.floor(pos.y) + 2,
                Math.floor(pos.z)
            );
        }
        ctx.setPhase(3, 1, 'complete');
        return { sealed: !!sealBlock };
    }
}

class PillarUpSkill extends SkillContract {
    constructor() {
        super({
            name: 'pillar_up',
            behaviorId: 2,
            owner: 'script',
            priority: 2,
            timeoutMs: 8000,
            cooldownMs: 4000,
        });
    }

    checkPrecondition(ctx) {
        const obs = getLatestObs(ctx.agent);
        const blockCount = ctx.agent?.survivalPolicy?._quickCheck?.()?.placeableBlocks ?? 0;
        return !!ctx.bot?.entity && !obs?.inWater && blockCount >= 3;
    }

    checkSuccess(ctx, startObs) {
        const startY = ctx.startPosition?.y ?? ctx.bot?.entity?.position?.y ?? 0;
        const endY = ctx.bot?.entity?.position?.y ?? startY;
        const yDelta = endY - startY;
        return yDelta >= 2;
    }

    async execute(ctx) {
        const now = Date.now();
        const bot = ctx.bot;
        const bridge = bot?._bridge;
        if (!bridge) throw new Error('no_bridge');

        ctx.setTarget(null);
        ctx.setActionMask({
            forward: false, back: false, left: false, right: false,
            jump: true, sprint: false, attack: false, use: true, sneak: true, hotbar: true,
        });
        ctx.scheduler?.setConstraints?.({
            freezeMicro: true,
            freezeMicroUntil: now + 6000,
        });
        ctx.scheduler?.clearProposal?.('micro');

        // Find a placeable block in inventory
        const pillarBlocks = ['cobblestone', 'dirt', 'netherrack', 'stone', 'cobbled_deepslate',
            'deepslate', 'andesite', 'diorite', 'granite', 'sandstone'];
        const blockItem = bot.inventory?.items?.()?.find(i => pillarBlocks.includes(i.name));
        if (!blockItem) throw new Error('no_blocks');

        // Equip the block
        try { await bot.equip(blockItem, 'hand'); } catch (e) {}

        // Look straight down (pitch=90)
        try {
            await bridge.sendCommand('rl_action', {
                forward: false, back: false, left: false, right: false,
                jump: false, sprint: false, attack: false, use: false, sneak: false,
                dyaw: 0, dpitch: 90,
            });
        } catch (e) {}
        await new Promise(r => setTimeout(r, 200));

        const startY = bot.entity?.position?.y ?? 0;
        ctx.setPhase(1, 0.2, 'pillar');

        // Pillar up 3 blocks: jump + place block under feet via rl_action
        for (let i = 0; i < 3; i++) {
            // Jump phase — record demo: jump + look down
            const jumpAction = {
                forward: false, back: false, left: false, right: false,
                jump: true, sprint: false, attack: false, use: false, sneak: false,
                dyaw: 0, dpitch: 90,
            };
            try { await bridge.sendCommand('rl_action', jumpAction); } catch (e) {}
            writeSkillDemo(ctx.agent, jumpAction, 'fight');
            await new Promise(r => setTimeout(r, 350));

            // Place phase — record demo: use (place block) + sneak + look down
            const placeAction = {
                forward: false, back: false, left: false, right: false,
                jump: false, sprint: false, attack: false, use: true, sneak: true,
                dyaw: 0, dpitch: 90,
            };
            try { await bridge.sendCommand('rl_action', placeAction); } catch (e) {}
            writeSkillDemo(ctx.agent, placeAction, 'fight');
            await new Promise(r => setTimeout(r, 300));

            // Release phase
            try {
                await bridge.sendCommand('rl_action', {
                    forward: false, back: false, left: false, right: false,
                    jump: false, sprint: false, attack: false, use: false, sneak: false,
                    dyaw: 0, dpitch: 0,
                });
            } catch (e) {}
            await new Promise(r => setTimeout(r, 200));
        }

        const endY = bot.entity?.position?.y ?? startY;
        console.log(`[PillarUp] startY=${startY.toFixed(1)} endY=${endY.toFixed(1)} delta=${(endY-startY).toFixed(1)}`);
        ctx.setPhase(2, 1, 'complete');
        return { raised: true, delta: endY - startY };
    }
}

class SwimToShoreSkill extends SkillContract {
    constructor() {
        super({
            name: 'swim_to_shore',
            behaviorId: 3,
            owner: 'script',
            priority: 2,
            timeoutMs: 12000,
            cooldownMs: 3000,
        });
    }

    checkPrecondition(ctx) {
        const obs = getLatestObs(ctx.agent);
        return !!obs?.inWater;
    }

    checkSuccess(ctx) {
        const endObs = getLatestObs(ctx.agent);
        return !!endObs && !endObs.inWater && !!endObs.onGround;
    }

    async execute(ctx) {
        const skills = await import('./library/skills.js');
        const landPos = ctx.agent.microCtrl?.escapePlanner?._lastLandPos || null;
        if (!landPos) throw new Error('no_land_memory');
        ctx.setTarget({ x: landPos.x, y: landPos.y ?? 80, z: landPos.z });
        ctx.setActionMask({
            forward: true, back: true, left: true, right: true,
            jump: true, sprint: true, attack: false, use: false, sneak: false, hotbar: false,
        });
        ctx.setPhase(1, 0.2, 'goto_land');
        ctx.recordAction({ type: 'script_call', name: 'goToPosition', args: { x: landPos.x, y: landPos.y ?? 80, z: landPos.z, closeness: 2 } }, { phaseId: 1, progress: 0.2, phaseLabel: 'goto_land', target: { x: landPos.x, y: landPos.y ?? 80, z: landPos.z } });
        await skills.goToPosition(ctx.bot, landPos.x, landPos.y ?? 80, landPos.z, 2);
        ctx.setPhase(2, 1, 'complete');
        return { landed: true };
    }
}

class HitAndRunSkill extends SkillContract {
    constructor() {
        super({
            name: 'hit_and_run',
            behaviorId: 4,
            owner: 'script',
            priority: 2,
            timeoutMs: 8000,
            cooldownMs: 3000,
        });
    }

    checkPrecondition(ctx) {
        const obs = getLatestObs(ctx.agent);
        return !!ctx.bot?.entity && (obs?.threats || []).length > 0;
    }

    checkSuccess(ctx, startObs) {
        const endObs = getLatestObs(ctx.agent);
        if (!startObs || !endObs) return false;
        const startDist = (startObs.threats || [])[0]?.dist ?? 99;
        const endDist = (endObs.threats || [])[0]?.dist ?? 99;
        return endDist > startDist || (endObs.hp ?? 20) >= (startObs.hp ?? 20);
    }

    async execute(ctx) {
        const skills = await import('./library/skills.js');
        const bot = ctx.bot;
        const bridge = bot?._bridge;
        if (!bridge) throw new Error('no_bridge');
        const state = ctx.agent.survivalPolicy?._quickCheck?.();
        const mobType = state?.closestMobType;
        const target = Object.values(bot.entities || {}).find(e =>
            mobType && e.name?.includes(mobType) && e !== bot.entity
        );
        if (!target) throw new Error('no_target');
        ctx.setTarget({ id: target.id, name: mobType });
        ctx.setPhase(1, 0.2, 'engage');

        // Equip best weapon (inline — equipHighestAttack is not exported from skills)
        let weapons = bot.inventory.items().filter(item => item.name.includes('sword') || (item.name.includes('axe') && !item.name.includes('pickaxe')));
        if (weapons.length === 0)
            weapons = bot.inventory.items().filter(item => item.name.includes('pickaxe') || item.name.includes('shovel'));
        if (weapons.length > 0) {
            weapons.sort((a, b) => (b.attackDamage || 0) - (a.attackDamage || 0));
            await bot.equip(weapons[0], 'hand');
        }

        // Record fight demos during engage
        const demoInterval = setInterval(() => {
            const obs = getLatestObs(ctx.agent);
            const t = obs?.threats?.[0];
            const dist = t?.dist ?? 99;
            writeSkillDemo(ctx.agent, {
                forward: dist > 3, back: dist < 2,
                left: false, right: false,
                jump: false, sprint: dist > 3,
                attack: dist <= 3 && (obs?.attackCooldown || 0) >= 0.9,
                use: false, sneak: false,
                dyaw: 0, dpitch: 0,
            }, 'fight');
        }, 200);

        try {
            // Approach via Baritone goto (handles pathfinding around obstacles)
            const dist = bot.entity.position.distanceTo(target.position);
            if (dist >= 4 && mobType !== 'creeper') {
                try {
                    await bridge.sendCommand('goto', {
                        x: target.position.x, y: target.position.y, z: target.position.z, range: 3
                    }, 4000);
                } catch (e) {}
            }
            // Attack via bridge (proper MC attack with swingtimer)
            try {
                await bridge.sendCommand('attack', { entityId: target.id });
            } catch (e) {}
            await new Promise(r => setTimeout(r, 500));
        } finally {
            clearInterval(demoInterval);
        }

        // Retreat: sprint backward
        ctx.setPhase(2, 0.8, 'retreat');
        for (let i = 0; i < 8; i++) {
            const retreatAction = {
                forward: false, back: true, left: false, right: false,
                jump: false, sprint: true, attack: false, use: false, sneak: false,
                dyaw: 0, dpitch: 0,
            };
            try { await bridge.sendCommand('rl_action', retreatAction); } catch (e) {}
            writeSkillDemo(ctx.agent, retreatAction, 'fight');
            await new Promise(r => setTimeout(r, 100));
        }

        console.log(`[HitAndRun] Completed vs ${mobType}`);
        ctx.setPhase(3, 1, 'complete');
        return { target: mobType };
    }
}

export function registerBuiltinSurvivalSkills(registry) {
    registry.register(new DigShelterSkill());
    registry.register(new PillarUpSkill());
    registry.register(new SwimToShoreSkill());
    registry.register(new HitAndRunSkill());
    return registry;
}
