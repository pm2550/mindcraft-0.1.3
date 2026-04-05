package com.mindcraft.bridge;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.network.ClientPlayerEntity;
import net.minecraft.entity.Entity;
import net.minecraft.entity.LivingEntity;
import net.minecraft.entity.mob.HostileEntity;
import net.minecraft.entity.passive.AnimalEntity;
import net.minecraft.entity.player.PlayerEntity;
import net.minecraft.item.ItemStack;
import net.minecraft.registry.Registries;
import net.minecraft.util.Identifier;
import net.minecraft.util.math.BlockPos;
import net.minecraft.client.world.ClientWorld;
import net.minecraft.entity.EquipmentSlot;
import net.minecraft.world.World;
import baritone.api.BaritoneAPI;
import baritone.api.IBaritone;
import baritone.api.behavior.IPathingBehavior;
import baritone.api.utils.IInputOverrideHandler;
import baritone.api.utils.input.Input;
import baritone.api.pathing.goals.Goal;
import baritone.api.pathing.goals.GoalXZ;
import baritone.api.pathing.goals.GoalComposite;
import baritone.api.pathing.path.IPathExecutor;
import baritone.api.utils.interfaces.IGoalRenderPos;

/**
 * 状态报告器 - 收集游戏状态并序列化为 JSON
 */
public class StateReporter {

    /**
     * 构建玩家状态 JSON
     */
    public String buildStateJson(MinecraftClient client) {
        ClientPlayerEntity player = client.player;
        if (player == null) return "{}";

        JsonObject json = new JsonObject();
        json.addProperty("type", "state");
        json.addProperty("timestamp", System.currentTimeMillis());

        // 位置
        JsonObject pos = new JsonObject();
        pos.addProperty("x", Math.round(player.getX() * 100.0) / 100.0);
        pos.addProperty("y", Math.round(player.getY() * 100.0) / 100.0);
        pos.addProperty("z", Math.round(player.getZ() * 100.0) / 100.0);
        json.add("position", pos);

        json.addProperty("yaw", player.getYaw());
        json.addProperty("pitch", player.getPitch());

        // 生命值 & 食物
        json.addProperty("health", player.getHealth());
        json.addProperty("maxHealth", player.getMaxHealth());
        json.addProperty("food", player.getHungerManager().getFoodLevel());
        json.addProperty("saturation", player.getHungerManager().getSaturationLevel());
        json.addProperty("oxygen", player.getAir());

        // 世界信息
        World world = client.world;
        json.addProperty("dimension", world.getRegistryKey().getValue().toString());
        json.addProperty("gameMode", client.interactionManager != null ?
                client.interactionManager.getCurrentGameMode().name().toLowerCase(java.util.Locale.ROOT) : "unknown");
        json.addProperty("onGround", player.isOnGround());
        json.addProperty("isInWater", player.isTouchingWater());
        json.addProperty("timeOfDay", world.getTimeOfDay() % 24000);
        json.addProperty("isRaining", world.isRaining());
        json.addProperty("isThundering", world.isThundering());

        // 经验
        json.addProperty("xpLevel", player.experienceLevel);
        json.addProperty("xpProgress", player.experienceProgress);

        // 当前持有物品
        ItemStack held = player.getMainHandStack();
        if (!held.isEmpty()) {
            json.addProperty("heldItem", Registries.ITEM.getId(held.getItem()).getPath());
            json.addProperty("heldItemCount", held.getCount());
        }

        // 用户名
        json.addProperty("username", player.getName().getString());

        return json.toString();
    }

    /**
     * 构建附近实体 JSON
     */
    public String buildEntitiesJson(MinecraftClient client) {
        ClientPlayerEntity player = client.player;
        if (player == null) return "{}";

        JsonObject json = new JsonObject();
        json.addProperty("type", "entities");

        JsonArray entities = new JsonArray();
        double maxRange = 48.0;

        for (Entity entity : client.world.getEntities()) {
            if (entity == player) continue;
            double dist = entity.distanceTo(player);
            if (dist > maxRange) continue;

            JsonObject e = new JsonObject();
            e.addProperty("id", entity.getId());

            // 获取实体类型名
            Identifier entityId = Registries.ENTITY_TYPE.getId(entity.getType());
            e.addProperty("name", entityId.getPath());

            // 分类
            String entityType = "other";
            if (entity instanceof PlayerEntity) entityType = "player";
            else if (entity instanceof HostileEntity) entityType = "hostile";
            else if (entity instanceof AnimalEntity) entityType = "animal";
            e.addProperty("entityType", entityType);

            // 如果是玩家，用玩家名
            if (entity instanceof PlayerEntity pe) {
                e.addProperty("displayName", pe.getName().getString());
            }

            e.addProperty("x", Math.round(entity.getX() * 10.0) / 10.0);
            e.addProperty("y", Math.round(entity.getY() * 10.0) / 10.0);
            e.addProperty("z", Math.round(entity.getZ() * 10.0) / 10.0);
            e.addProperty("distance", Math.round(dist * 10.0) / 10.0);

            if (entity instanceof LivingEntity le) {
                e.addProperty("health", le.getHealth());
                e.addProperty("maxHealth", le.getMaxHealth());
            }

            entities.add(e);
        }

        json.add("entities", entities);
        return json.toString();
    }

    /**
     * 构建背包 JSON
     */
    public String buildInventoryJson(MinecraftClient client) {
        ClientPlayerEntity player = client.player;
        if (player == null) return "{}";

        JsonObject json = new JsonObject();
        json.addProperty("type", "inventory");

        JsonArray slots = new JsonArray();
        for (int i = 0; i < player.getInventory().size(); i++) {
            ItemStack stack = player.getInventory().getStack(i);
            if (stack.isEmpty()) continue;

            JsonObject item = new JsonObject();
            item.addProperty("slot", i);
            item.addProperty("item", Registries.ITEM.getId(stack.getItem()).getPath());
            item.addProperty("count", stack.getCount());

            if (stack.isDamageable()) {
                item.addProperty("durability", stack.getMaxDamage() - stack.getDamage());
                item.addProperty("maxDurability", stack.getMaxDamage());
            }

            slots.add(item);
        }
        json.add("slots", slots);

        // 盔甲
        JsonObject armor = new JsonObject();
        String[] armorSlots = {"boots", "leggings", "chestplate", "helmet"};
        for (int i = 0; i < 4; i++) {
            EquipmentSlot[] armorEquipSlots = {EquipmentSlot.FEET, EquipmentSlot.LEGS, EquipmentSlot.CHEST, EquipmentSlot.HEAD};
            ItemStack armorStack = player.getEquippedStack(armorEquipSlots[i]);
            if (!armorStack.isEmpty()) {
                JsonObject a = new JsonObject();
                a.addProperty("item", Registries.ITEM.getId(armorStack.getItem()).getPath());
                if (armorStack.isDamageable()) {
                    a.addProperty("durability", armorStack.getMaxDamage() - armorStack.getDamage());
                }
                armor.add(armorSlots[i], a);
            }
        }
        json.add("armor", armor);

        // 副手
        ItemStack offhand = player.getOffHandStack();
        if (!offhand.isEmpty()) {
            JsonObject oh = new JsonObject();
            oh.addProperty("item", Registries.ITEM.getId(offhand.getItem()).getPath());
            oh.addProperty("count", offhand.getCount());
            json.add("offhand", oh);
        }

        json.addProperty("selectedSlot", player.getInventory().getSelectedSlot());

        return json.toString();
    }

    /**
     * 构建聊天消息 JSON
     */
    public String buildChatJson(String message, boolean isWhisper) {
        JsonObject json = new JsonObject();
        json.addProperty("type", "chat");
        json.addProperty("message", message);
        json.addProperty("isWhisper", isWhisper);
        return json.toString();
    }

    public String buildChatJson(String sender, String message, boolean isWhisper) {
        JsonObject json = new JsonObject();
        json.addProperty("type", "chat");
        json.addProperty("sender", sender);
        json.addProperty("message", message);
        json.addProperty("isWhisper", isWhisper);
        return json.toString();
    }

    /**
     * RL Observation: compact, tick-level state for RL policy training.
     * Pushed every tick (50ms) when RL mode is active.
     */
    public String buildRLObservation(MinecraftClient client) {
        ClientPlayerEntity player = client.player;
        if (player == null) return null;

        JsonObject obs = new JsonObject();
        obs.addProperty("type", "rl_obs");
        obs.addProperty("tick", client.world.getTime());

        // === Self ===
        obs.addProperty("hp", Math.round(player.getHealth() * 10) / 10.0);
        obs.addProperty("food", player.getHungerManager().getFoodLevel());
        obs.addProperty("armor", player.getArmor());
        obs.addProperty("yaw", Math.round(player.getYaw() * 10) / 10.0);
        obs.addProperty("pitch", Math.round(player.getPitch() * 10) / 10.0);
        obs.addProperty("onGround", player.isOnGround());
        obs.addProperty("px", Math.round(player.getX() * 10) / 10.0);
        obs.addProperty("pz", Math.round(player.getZ() * 10) / 10.0);
        obs.addProperty("inWater", player.isTouchingWater());
        obs.addProperty("sprinting", player.isSprinting());

        // Velocity
        obs.addProperty("vx", Math.round(player.getVelocity().x * 100) / 100.0);
        obs.addProperty("vy", Math.round(player.getVelocity().y * 100) / 100.0);
        obs.addProperty("vz", Math.round(player.getVelocity().z * 100) / 100.0);

        // Held item + attack cooldown
        ItemStack held = player.getMainHandStack();
        obs.addProperty("held", held.isEmpty() ? "empty" : Registries.ITEM.getId(held.getItem()).getPath());
        obs.addProperty("selectedSlot", player.getInventory().getSelectedSlot());
        obs.addProperty("attackCooldown", player.getAttackCooldownProgress(0.0f));

        // === Threats: nearest 5 hostile mobs as relative vectors ===
        JsonArray threats = new JsonArray();
        double px = player.getX(), py = player.getY(), pz = player.getZ();
        float playerYaw = player.getYaw();

        for (Entity entity : client.world.getEntities()) {
            if (entity == player) continue;
            if (!(entity instanceof HostileEntity)) continue;
            double dist = entity.distanceTo(player);
            if (dist > 24) continue;

            JsonObject t = new JsonObject();
            double dx = entity.getX() - px;
            double dy = entity.getY() - py;
            double dz = entity.getZ() - pz;

            // Relative to player facing direction
            double radYaw = Math.toRadians(playerYaw);
            double relX = dx * Math.cos(radYaw) + dz * Math.sin(radYaw);  // right(+) / left(-)
            double relZ = -dx * Math.sin(radYaw) + dz * Math.cos(radYaw); // front(+) / back(-)

            t.addProperty("type", Registries.ENTITY_TYPE.getId(entity.getType()).getPath());
            t.addProperty("dx", Math.round(relX * 10) / 10.0);  // relative right/left
            t.addProperty("dz", Math.round(relZ * 10) / 10.0);  // relative front/back
            t.addProperty("dy", Math.round(dy * 10) / 10.0);
            t.addProperty("dist", Math.round(dist * 10) / 10.0);
            t.addProperty("hp", entity instanceof LivingEntity ? Math.round(((LivingEntity) entity).getHealth() * 10) / 10.0 : -1);

            threats.add(t);
            if (threats.size() >= 5) break;
        }
        obs.add("threats", threats);

        // === 3D Terrain scan: 51x51 base grid × 7 vertical layers (y-2 to y+4) ===
        // Layer 0-6: y offsets [-2, -1, 0, 1, 2, 3, 4] relative to player feet.
        // Each cell: 0=air/passable, 1=solid, 2=liquid
        // This gives the RL policy height information — it can see canyon walls,
        // 1-block steps vs 3-block cliffs, overhead cover, etc.
        // Total: 51 * 51 * 7 = 18207 values.
        // For backward compat, also emit the old 2D grid as "terrain" (2601 values).
        ClientWorld world = client.world;
        BlockPos playerPos = player.getBlockPos();

        // Old 2D terrain (backward compatible — escape planner still uses this)
        JsonArray terrain = new JsonArray();
        for (int dz2 = -25; dz2 <= 25; dz2++) {
            for (int dx2 = -25; dx2 <= 25; dx2++) {
                BlockPos feet = playerPos.add(dx2, 0, dz2);
                BlockPos head = playerPos.add(dx2, 1, dz2);
                BlockPos below = playerPos.add(dx2, -1, dz2);

                boolean headClear = world.getBlockState(head).isAir();
                boolean groundSolid = !world.getBlockState(below).isAir();

                int code;
                if (!world.getBlockState(feet).getFluidState().isEmpty() && !world.getBlockState(feet).isAir()) {
                    code = 2;
                } else if (headClear && (world.getBlockState(feet).isAir() || world.getBlockState(feet).isReplaceable()) && groundSolid) {
                    code = 0;
                } else if (world.getBlockState(feet).isAir() && !groundSolid) {
                    code = 3;
                } else {
                    code = 1;
                }
                terrain.add(code);
            }
        }
        obs.add("terrain", terrain); // 2601 values, 51x51 grid (backward compat)

        // === 3D Terrain: heightmap + column encoding (smart compression) ===
        // Instead of raw 3D voxel grid (21x21x7 = 3087 values), we send:
        //   1. heightmap[21x21=441]: relative ground height at each (dx,dz) position.
        //      Value = highest solid block's y offset from player feet (-8 to +8, clamped).
        //      This tells RL: "there's a 3-block wall to the left" or "2-block drop ahead".
        //   2. ceilmap[21x21=441]: ceiling height (lowest solid above ground).
        //      Useful for: caves, overhangs, tunnels. Value 0 = no ceiling in range.
        //   3. liquid flags packed into heightmap: negative height = liquid at surface.
        // Total: 882 values. Smaller than old 2D grid (2601) yet carries full 3D info.

        int hmRange = 25;  // ±25 blocks = full 50m view (same as old 2D grid)
        int scanUp = 8;    // scan up to 8 blocks above feet
        int scanDown = 6;  // scan down to 6 blocks below feet
        JsonArray heightmap = new JsonArray();
        JsonArray ceilmap = new JsonArray();

        for (int dz2 = -hmRange; dz2 <= hmRange; dz2++) {
            for (int dx2 = -hmRange; dx2 <= hmRange; dx2++) {
                // Find ground height: scan down from player feet to find highest solid
                int groundY = -scanDown - 1; // default: deep void
                for (int dy = scanUp; dy >= -scanDown; dy--) {
                    BlockPos check = playerPos.add(dx2, dy, dz2);
                    var st = world.getBlockState(check);
                    if (!st.isAir() && !st.isReplaceable() && st.getFluidState().isEmpty()) {
                        groundY = dy;
                        break;
                    }
                }

                // Check if surface is liquid
                boolean isLiquid = false;
                if (groundY > -scanDown - 1) {
                    BlockPos surfaceAbove = playerPos.add(dx2, groundY + 1, dz2);
                    if (!world.getBlockState(surfaceAbove).getFluidState().isEmpty()) {
                        isLiquid = true;
                    }
                } else {
                    // No solid found — check if it's all liquid
                    BlockPos feetLevel = playerPos.add(dx2, 0, dz2);
                    if (!world.getBlockState(feetLevel).getFluidState().isEmpty()) {
                        isLiquid = true;
                        groundY = -1; // liquid surface roughly at feet
                    }
                }

                // Encode: positive = solid ground height, negative = liquid surface
                int heightVal = Math.max(-12, Math.min(12, groundY));
                if (isLiquid) heightVal = -Math.abs(heightVal) - 1; // ensure negative for liquid
                heightmap.add(heightVal);

                // Find ceiling: lowest solid above ground level
                int ceilY = 0; // 0 = no ceiling
                if (groundY > -scanDown - 1) {
                    for (int dy = groundY + 2; dy <= groundY + scanUp; dy++) {
                        BlockPos check = playerPos.add(dx2, dy, dz2);
                        var st = world.getBlockState(check);
                        if (!st.isAir() && !st.isReplaceable() && st.getFluidState().isEmpty()) {
                            ceilY = dy - groundY; // relative to ground
                            break;
                        }
                    }
                }
                ceilmap.add(ceilY);
            }
        }
        obs.add("heightmap", heightmap); // 2601 values, 51x51 (±25 range, full resolution)
        obs.add("ceilmap", ceilmap);     // 2601 values, 51x51

        // === Cover detection: LOS to nearest threat (always present in schema) ===
        boolean hasLOS = false; // default: no threat = no LOS
        if (threats.size() > 0) {
            Entity nearest = null;
            double nearestDist = 999;
            for (Entity entity : client.world.getEntities()) {
                if (entity == player || !(entity instanceof HostileEntity)) continue;
                double d = entity.distanceTo(player);
                if (d < nearestDist) { nearestDist = d; nearest = entity; }
            }
            if (nearest != null) {
                hasLOS = true;
                for (int step = 1; step <= 3; step++) {
                    double t = step / 4.0;
                    double sx = px + (nearest.getX() - px) * t;
                    double sy = py + 1.5 + (nearest.getEyeY() - py - 1.5) * t;
                    double sz = pz + (nearest.getZ() - pz) * t;
                    if (!world.getBlockState(new BlockPos((int)sx, (int)sy, (int)sz)).isAir()) {
                        hasLOS = false;
                        break;
                    }
                }
            }
        }
        obs.addProperty("los", hasLOS);

        // === Can dig ===
        BlockPos belowPlayer = playerPos.down();
        obs.addProperty("canDigDown", !world.getBlockState(belowPlayer).isAir() && world.getBlockState(belowPlayer).getHardness(world, belowPlayer) >= 0);

        // === Baritone state: goal, path progress, input overrides ===
        try {
            IBaritone baritone = BaritoneAPI.getProvider().getPrimaryBaritone();
            IPathingBehavior pathing = baritone.getPathingBehavior();
            IInputOverrideHandler input = baritone.getInputOverrideHandler();

            JsonObject bari = new JsonObject();
            bari.addProperty("pathing", pathing.isPathing());

            // Goal position (player-relative) — supports single goals and GoalComposite (mine)
            Goal goal = pathing.getGoal();
            // For GoalComposite: find the nearest sub-goal with a position
            if (goal instanceof GoalComposite gc) {
                double bestDist = Double.MAX_VALUE;
                BlockPos bestPos = null;
                for (Goal sub : gc.goals()) {
                    if (sub instanceof IGoalRenderPos sgr) {
                        BlockPos sp = sgr.getGoalPos();
                        double d = Math.sqrt(Math.pow(sp.getX() - px, 2) + Math.pow(sp.getZ() - pz, 2));
                        if (d < bestDist) { bestDist = d; bestPos = sp; }
                    }
                }
                if (bestPos != null) {
                    double gdx = bestPos.getX() - px, gdz = bestPos.getZ() - pz, gdy = bestPos.getY() - py;
                    double radYaw = Math.toRadians(playerYaw);
                    bari.addProperty("goalDx", Math.round((gdx * Math.cos(radYaw) + gdz * Math.sin(radYaw)) * 10) / 10.0);
                    bari.addProperty("goalDz", Math.round((-gdx * Math.sin(radYaw) + gdz * Math.cos(radYaw)) * 10) / 10.0);
                    bari.addProperty("goalDy", Math.round(gdy * 10) / 10.0);
                    bari.addProperty("goalDist", Math.round(bestDist * 10) / 10.0);
                }
            } else if (goal instanceof IGoalRenderPos grp) {
                BlockPos gp = grp.getGoalPos();
                double gdx = gp.getX() - px, gdz = gp.getZ() - pz, gdy = gp.getY() - py;
                double radYaw = Math.toRadians(playerYaw);
                bari.addProperty("goalDx", Math.round((gdx * Math.cos(radYaw) + gdz * Math.sin(radYaw)) * 10) / 10.0);
                bari.addProperty("goalDz", Math.round((-gdx * Math.sin(radYaw) + gdz * Math.cos(radYaw)) * 10) / 10.0);
                bari.addProperty("goalDy", Math.round(gdy * 10) / 10.0);
                bari.addProperty("goalDist", Math.round(Math.sqrt(gdx * gdx + gdz * gdz) * 10) / 10.0);
            } else if (goal instanceof GoalXZ gxz) {
                double gdx = gxz.getX() - px, gdz = gxz.getZ() - pz;
                double radYaw = Math.toRadians(playerYaw);
                bari.addProperty("goalDx", Math.round((gdx * Math.cos(radYaw) + gdz * Math.sin(radYaw)) * 10) / 10.0);
                bari.addProperty("goalDz", Math.round((-gdx * Math.sin(radYaw) + gdz * Math.cos(radYaw)) * 10) / 10.0);
                bari.addProperty("goalDy", 0);
                bari.addProperty("goalDist", Math.round(Math.sqrt(gdx * gdx + gdz * gdz) * 10) / 10.0);
            }

            // Estimated ticks to goal
            pathing.estimatedTicksToGoal().ifPresent(t -> bari.addProperty("estTicks", Math.round(t)));

            // Path progress (0.0 - 1.0)
            IPathExecutor current = pathing.getCurrent();
            if (current != null && current.getPath() != null) {
                int pathLen = current.getPath().length();
                if (pathLen > 0) {
                    bari.addProperty("pathProgress", Math.round(current.getPosition() * 100.0 / pathLen) / 100.0);
                }
            }

            // 9 Baritone input overrides — exact keys Baritone forces each tick
            bari.addProperty("inFwd", input.isInputForcedDown(Input.MOVE_FORWARD));
            bari.addProperty("inBack", input.isInputForcedDown(Input.MOVE_BACK));
            bari.addProperty("inLeft", input.isInputForcedDown(Input.MOVE_LEFT));
            bari.addProperty("inRight", input.isInputForcedDown(Input.MOVE_RIGHT));
            bari.addProperty("inJump", input.isInputForcedDown(Input.JUMP));
            bari.addProperty("inSneak", input.isInputForcedDown(Input.SNEAK));
            bari.addProperty("inSprint", input.isInputForcedDown(Input.SPRINT));
            bari.addProperty("inAttack", input.isInputForcedDown(Input.CLICK_LEFT));
            bari.addProperty("inUse", input.isInputForcedDown(Input.CLICK_RIGHT));

            // Player state for imitation learning
            bari.addProperty("swinging", player.handSwinging);
            bari.addProperty("sneaking", player.isSneaking());

            obs.add("baritone", bari);
        } catch (Exception e) {
            // Baritone may not be loaded; skip silently
        }

        // === Actual key states: captures ALL sources (Baritone, scripts, mineflayer, RL) ===
        obs.addProperty("keyFwd", client.options.forwardKey.isPressed());
        obs.addProperty("keyBack", client.options.backKey.isPressed());
        obs.addProperty("keyLeft", client.options.leftKey.isPressed());
        obs.addProperty("keyRight", client.options.rightKey.isPressed());
        obs.addProperty("keyJump", client.options.jumpKey.isPressed());
        obs.addProperty("keySneak", client.options.sneakKey.isPressed());
        obs.addProperty("keySprint", client.options.sprintKey.isPressed());
        obs.addProperty("keyAttack", client.options.attackKey.isPressed());
        obs.addProperty("keyUse", client.options.useKey.isPressed());
        obs.addProperty("hotbar", player.getInventory().getSelectedSlot());

        return obs.toString();
    }
}
