const settings = {
    "minecraft_version": "1.21.11",
    // "host": "127.0.0.1", // or "localhost", "your.ip.address.here"
    "host": "192.9.134.169",
    "port": 25566,
    "auth": "microsoft", // or "offline"

    // === Fabric Bridge 模式 ===
    // 设为 "fabric" 使用真实 MC 客户端 + Baritone（需要运行 Fabric Mod）
    // 设为 "mineflayer" 使用原版 mineflayer 协议客户端
    "backend": "fabric",
    "fabric_ws_host": "localhost",
    "fabric_ws_port": 8899,

    // the mindserver manages all agents and hosts the UI
    "mindserver_port": 8081,
    "auto_open_ui": true, // opens UI in browser on startup
    
    "base_profile": "assistant", // survival, assistant, creative, or god_mode
    "profiles": [
        "./andy.json",       // GaoBot1
        // "./GaoBot2.json",    // GaoBot2
        // "./GaoBot3.json",    // GaoBot3
    ],

    "load_memory": false, // load memory from previous session
    "init_message": "!goal(\"Step 1: !baritone(goto 1200 80 0). Step 2: !baritone(mine oak_log) to get wood. Step 3: !craftRecipe(oak_planks 1) then !craftRecipe(crafting_table 1) then !craftRecipe(stick 1) then !craftRecipe(wooden_sword 1) then !craftRecipe(wooden_pickaxe 1). Step 4: !baritone(mine stone) for cobblestone. Step 5: Kill animals with sword for food. IMPORTANT: Use !craftRecipe not !baritone(mine) for crafting.\")", // sends to all on spawn
    "only_chat_with": [], // 只听这些玩家的指令，其他人说话会被忽略。留空则响应所有人
    // ===== 空闲时默认任务 =====
    // 当机器人空闲超过 idle_timeout 秒后，自动开始执行对应的默认任务
    // 用户可以随时用命令覆盖这些任务
    "idle_timeout": 20, // 空闲多少秒后开始默认任务（0 = 禁用）
    "agent_start_delay_ms": 4000, // delay between starting bots to avoid server rate limits
    "agent_restart_delay_ms": 5000, // delay before restarting a bot after disconnect
    "spawn_timeout_ms": 0, // 0 = no timeout, wait forever for Fabric mod to connect
    "spawn_warmup_ms": 5000, // delay before enabling behaviors after spawn (ms)
    "max_active_bots": 3, // limit how many bots start at once (0 = all)
    "enable_extended_modes": false, // enable extra auto behaviors (drowned hunting/food/gear/home storage)
    "enable_extra_plugins": false, // enable extra plugins like movement/gui
    "enable_armor_manager": false, // enable armor manager plugin (can be unstable)
    "enable_tool_plugin": true, // enable mineflayer-tool plugin (auto tool selection)
    "restart_on_stuck_action": false, // restart bot if an action can't be stopped (true) or reset state (false)
    "restart_after_smelt": false, // restart bot after smelting to refresh inventory (usually unnecessary)
    "allow_newaction_without_keyword": false, // allow newAction without explicit user keyword
    "newaction_keywords": ["newaction", "写代码", "新动作"], // keywords required to allow newAction
    "block_path_in_protected_areas": false, // whether bots can path into protected areas (false = allow)

    "speak": false,
    // allows all bots to speak through text-to-speech. 
    // specify speech model inside each profile with format: {provider}/{model}/{voice}.
    // if set to "system" it will use basic system text-to-speech. 
    // Works on windows and mac, but linux requires you to `apt install espeak`.

    "chat_ingame": true, // bot responses are shown in minecraft chat
    "language": "zh", // translate to/from this language. Use "zh" for Chinese. Supports these language names: https://cloud.google.com/translate/docs/languages
    "render_bot_view": false, // show bot's view in browser at localhost:3000, 3001...

    "allow_insecure_coding": true, // allows newAction command and model can write/run code on your computer. enable at own risk
    "allow_hand_dig": true, // allow breaking blocks above you without proper tools (slow, may drop nothing)
    "allow_vision": false, // allows vision model to interpret screenshots as inputs
    "blocked_actions" : ["!checkBlueprint", "!checkBlueprintLevel", "!getBlueprint", "!getBlueprintLevel", "!goal", "!endGoal"] , // block goal/endGoal to prevent LLM from overriding survival goal
    
    // ===== 安全护栏配置 =====
    "use_safeguard": true, // 启用安全护栏，只允许白名单命令
    "safeguard_whitelist": [
        // 查询类（安全，只读取信息）
        "!stats", "!inventory", "!nearbyBlocks", "!craftable", "!entities",
        "!modes", "!savedPlaces", "!getCraftingPlan", "!checkPrereqs", "!searchWiki", "!help",
        
        // 移动类（安全，只是移动）
        "!goToPlayer", "!followPlayer", "!goToCoordinates", "!searchForBlock",
        "!searchForEntity", "!moveAway", "!goToRememberedPlace", "!goToBed",
        
        // 控制类（安全，用于停止/控制）
        "!stop", "!stfu", "!restart", "!clearChat", "!stay", "!setMode",
        "!goal", "!endGoal",
        
        // 采集/制作/挖矿类（安全，正常游戏行为）
        "!collectBlocks", "!craftRecipe", "!smeltItem", "!clearFurnace",
        "!digDown", // 自带安全机制：遇到岩浆/水/4格落差会自动停止
        
        // 物品交互类（相对安全）
        "!givePlayer", "!consume", "!equip", "!putInChest", "!takeFromChest", 
        "!viewChest", "!rememberHere",
        
        // 战斗类（只打怪，不打玩家）
        "!attack",
        
        // 交易类
        "!showVillagerTrades", "!tradeWithVillager",
        
        // 视觉类
        "!lookAtPlayer", "!lookAtPosition",
        
        // 放置类（单个方块，相对安全）
        "!placeHere", "!useOn",

        // Baritone 自动化（导航、挖矿、探索）
        "!baritone",

        // 挖掘
        "!digUp",

        // 自定义动作（已允许）
        "!newAction",
        
        // Bot 对话
        "!startConversation", "!endConversation"
    ],
    // 危险命令（已被排除）:
    // "!newAction"     - 可执行任意代码，极其危险
    // "!attackPlayer"  - 攻击玩家，可能惹怒别人
    // "!discard"       - 可能丢弃重要物品
    
    // ===== 家的位置配置 =====
    // ⚠️ 把下面的坐标改成你家的实际坐标！按 F3 查看
    // 这个位置用于：1) 存放贵重物品 2) 保护区域中心
    "home_position": { "x": 99, "y": 108, "z": 2.4 }, // 格式: { "x": 100, "y": 64, "z": 200 } 或 null 表示未设置
    
    // ===== 贵重物品自动存放配置 =====
    "valuable_items": {
        "diamond": 3,      // 钻石超过3个就回家存
        "emerald": 5,      // 绿宝石超过5个就回家存
        "gold_ingot": 16,  // 金锭超过16个就回家存
        "netherite_ingot": 1, // 下界合金1个就回家存
        "netherite_scrap": 2  // 下界合金碎片2个就回家存
    },
    
    // ===== 防拆家：保护区域配置 =====
    // 在这些区域内，bot 不会破坏方块，会提示去更远的地方
    // 格式: { name: "区域名", x: 中心X, y: 中心Y, z: 中心Z, radius: 半径 }
    // ⚠️ 把下面的 x, z 改成你家的实际坐标！按 F3 查看
    "protected_areas": [
        // { "name": "家", "x": 99, "y": 108, "z": 2.4, "radius": 30 }
        // { "name": "我的家", "x": 你的X坐标, "y": 64, "z": 你的Z坐标, "radius": 50 }
    ],
    
    "code_timeout_mins": -1, // minutes code is allowed to run. -1 for no timeout
    "relevant_docs_count": 5, // number of relevant code function docs to select for prompting. -1 for all

    "max_messages": 8, // reduced for 9B model — less context = faster responses
    "num_examples": 1, // reduced to save context space
    "max_commands": -1, // max number of commands that can be used in consecutive responses. -1 for no limit
    "show_command_syntax": "none", // "full", "shortened", or "none" - 设为 none 减少聊天消息
    "narrate_behavior": false, // 关闭自动行为叙述（不会说 "Picking up item!" 等）
    "chat_bot_messages": false, // 不公开 bot 之间的消息
    "verbose_commands": false, // 关闭命令执行的详细输出

    "block_place_delay": 0, // delay between placing blocks (ms) if using newAction. helps avoid bot being kicked by anti-cheat mechanisms on servers.
  
    "log_all_prompts": false, // log ALL prompts to file
}

export default settings;
