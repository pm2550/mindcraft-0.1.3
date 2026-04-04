/**
 * Local Escape Planner: replaces raw action mixing for flee/hide.
 *
 * Architecture:
 *   ThreatField → Candidate Polylines → Score → Best Path → Executor
 *
 * Key design decisions:
 * - Escape direction computed in WORLD SPACE (stable as player turns)
 * - Segments store world-space target angles
 * - Executor uses live yaw for heading error
 * - Long commit windows prevent primitive oscillation
 * - Dyaw capped proportional to current speed (no spinning in place)
 */

// ============================================================
// HELPERS
// ============================================================
function normalizeAngle(a) {
    a = a % 360;
    if (a > 180) a -= 360;
    if (a <= -180) a += 360;
    return a;
}

/** Convert player-relative (relX, relZ) to world-space (dx, dz). */
function relToWorld(relX, relZ, yawDeg) {
    const rad = yawDeg * Math.PI / 180;
    const c = Math.cos(rad), s = Math.sin(rad);
    return {
        dx: relX * c - relZ * s,
        dz: relX * s + relZ * c,
    };
}

/** Terrain grid index for world-space offset (dx, dz) clamped to [-25,25]. 51x51 grid. */
function terrainIdx(worldDx, worldDz) {
    const ix = Math.max(-25, Math.min(25, Math.round(worldDx)));
    const iz = Math.max(-25, Math.min(25, Math.round(worldDz)));
    return (iz + 25) * 51 + (ix + 25);
}

/** Convert world-space (dx, dz) to player-relative (relX, relZ). */
function worldToRel(dx, dz, yawDeg) {
    const rad = yawDeg * Math.PI / 180;
    const c = Math.cos(rad), s = Math.sin(rad);
    return {
        x: dx * c + dz * s,
        z: -dx * s + dz * c,
    };
}

/** MC forward world direction from yaw. */
function yawToWorldDir(yawDeg) {
    const rad = yawDeg * Math.PI / 180;
    return { x: -Math.sin(rad), z: Math.cos(rad) };
}

/** Estimate nearby shore direction from 51x51 terrain (world-aligned). */
function estimateShoreWorldDir(terrain) {
    if (!terrain || terrain.length < 2601) return null;
    let sx = 0, sz = 0, wsum = 0;
    for (let dz = -25; dz <= 25; dz++) {
        for (let dx = -25; dx <= 25; dx++) {
            if (dx === 0 && dz === 0) continue;
            const idx = (dz + 25) * 51 + (dx + 25);
            if (terrain[idx] !== 0) continue;
            const d2 = dx * dx + dz * dz;
            const w = 1 / Math.max(1, d2);
            sx += dx * w;
            sz += dz * w;
            wsum += w;
        }
    }
    if (wsum <= 0) return null;
    const mag = Math.sqrt(sx * sx + sz * sz) || 1;
    return { x: sx / mag, z: sz / mag };
}

// ============================================================
// THREAT FIELD (world-space escape direction)
// ============================================================
class ThreatField {
    constructor() {
        this._prevThreats = null;
    }

    /**
     * Compute threat field. Returns escape direction in WORLD SPACE
     * so it's stable regardless of player facing.
     */
    compute(obs) {
        const threats = (obs?.threats || []).filter(t => t && typeof t.dist === 'number');
        if (threats.length === 0) {
            return { escapeDir: { x: 0, z: 1 }, escapeYaw: 0, threats: [], pressure: 0 };
        }

        const playerYaw = obs?.yaw || 0;

        const enriched = threats.map((t) => {
            let closingRate = 0;
            if (this._prevThreats) {
                const prevMatch = this._prevThreats
                    .filter(p => (p.type || '') === (t.type || ''))
                    .sort((a, b) => Math.abs((a.dist||32)-(t.dist||32)) - Math.abs((b.dist||32)-(t.dist||32)))[0];
                if (prevMatch) {
                    closingRate = ((prevMatch.dist || 32) - (t.dist || 32)) / 0.05;
                }
            }
            const type = t.type || 'unknown';
            const distWeight = 1 / Math.max(t.dist || 1, 0.5) ** 2;
            const speedMult = closingRate > 2 ? 2.0 : (closingRate > 0 ? 1.5 : 1.0);
            const typeMult = type.includes('creeper') ? 3 : 1;

            // Java sends player-relative dx/dz (rotated by playerYaw).
            // Convert to world-space for stable escape direction computation.
            const world = relToWorld(t.dx || 0, t.dz || 0, playerYaw);
            return {
                dx: t.dx || 0,   // player-relative (for primitive generation)
                dz: t.dz || 0,
                worldDx: world.dx, // world-space (for escapeYaw)
                worldDz: world.dz,
                dy: t.dy || 0,
                dist: t.dist || 32,
                hp: t.hp || 20,
                type,
                closingRate,
                isRanged: ['skeleton', 'stray', 'pillager', 'drowned'].some(r => type.includes(r)),
                isExplosive: type.includes('creeper'),
                isMelee: ['zombie', 'husk', 'spider', 'vindicator', 'ravager', 'drowned', 'piglin', 'hoglin'].some(r => type.includes(r)),
                engaged: (t.dist || 32) < 16 && closingRate > 0,
                weight: distWeight * speedMult * typeMult,
            };
        });

        this._prevThreats = threats;

        // Compute escape direction in WORLD space (stable as player turns)
        let wx = 0, wz = 0;
        for (const t of enriched) {
            wx += -(t.worldDx) * t.weight;
            wz += -(t.worldDz) * t.weight;
        }
        const mag = Math.sqrt(wx * wx + wz * wz) || 1;
        const pressure = enriched.reduce((s, t) => s + t.weight, 0);

        // World-space escape yaw (the MC yaw the player should face to flee)
        const escapeYaw = Math.atan2(-wx / mag, wz / mag) * (180 / Math.PI);
        // Note: MC forward at yaw θ is (-sin θ, cos θ) in world,
        // so yaw = atan2(-worldX, worldZ) * 180/PI

        // Also compute player-relative escape dir (for primitive generation/scoring)
        let ex = 0, ez = 0;
        for (const t of enriched) {
            ex += -(t.dx) * t.weight;
            ez += -(t.dz) * t.weight;
        }
        const relMag = Math.sqrt(ex * ex + ez * ez) || 1;

        return {
            escapeDir: { x: ex / relMag, z: ez / relMag }, // player-relative (for scoring)
            escapeYaw, // world-space yaw (stable target)
            threats: enriched,
            pressure,
        };
    }
}

// ============================================================
// MOTION PRIMITIVES
// ============================================================

function generatePrimitives(escapeDir, terrain) {
    const ex = escapeDir.x;
    const ez = escapeDir.z;

    return [
        {
            name: 'retreat_straight',
            segments: [{ dx: ex * 6, dz: ez * 6, ticks: 30 }],
        },
        {
            name: 'strafe_left',
            segments: [
                { dx: ex * 3 - 2, dz: ez * 3, ticks: 15 },
                { dx: ex * 3 - 2, dz: ez * 3, ticks: 15 },
            ],
        },
        {
            name: 'strafe_right',
            segments: [
                { dx: ex * 3 + 2, dz: ez * 3, ticks: 15 },
                { dx: ex * 3 + 2, dz: ez * 3, ticks: 15 },
            ],
        },
        {
            name: 'zigzag',
            segments: [
                { dx: ex * 2 - 2, dz: ez * 2, ticks: 10 },
                { dx: ex * 2 + 2, dz: ez * 2, ticks: 10 },
                { dx: ex * 2 - 2, dz: ez * 2, ticks: 10 },
            ],
        },
        {
            name: 'detour_left',
            segments: [
                { dx: -3, dz: 1, ticks: 10 },
                { dx: ex * 4, dz: ez * 4, ticks: 20 },
            ],
        },
        {
            name: 'detour_right',
            segments: [
                { dx: 3, dz: 1, ticks: 10 },
                { dx: ex * 4, dz: ez * 4, ticks: 20 },
            ],
        },
        {
            name: 'backpedal',
            segments: [{ dx: 0, dz: -4, ticks: 30 }],
        },
        {
            name: 'sprint_escape',
            segments: [{ dx: ex * 8, dz: ez * 8, ticks: 30 }],
        },
    ];
}

// ============================================================
// PATH SCORING
// ============================================================
function scorePath(primitive, threatField, terrain, playerYaw, constraints = null) {
    let score = 0;
    const { escapeDir, threats, pressure } = threatField;

    const totalDx = primitive.segments.reduce((s, seg) => s + seg.dx, 0);
    const totalDz = primitive.segments.reduce((s, seg) => s + seg.dz, 0);
    const pathMag = Math.sqrt(totalDx * totalDx + totalDz * totalDz) || 1;
    const alignment = (totalDx * escapeDir.x + totalDz * escapeDir.z) / pathMag;
    score += alignment * 3.0;

    score += pathMag * 0.1;

    const hasRanged = threats.some(t => t.isRanged && t.dist < 16);
    if (hasRanged && (primitive.name.includes('strafe') || primitive.name === 'zigzag')) {
        score += 1.5;
    }

    if (primitive.name === 'sprint_escape' || primitive.name === 'retreat_straight') {
        score += 0.5;
    }

    const hasCreeper = threats.some(t => t.isExplosive && t.dist < 8);
    if (hasCreeper && (primitive.name === 'sprint_escape' || primitive.name === 'retreat_straight')) {
        score += 3.0;
    }

    if (terrain && terrain.length >= 2601) {
        const normDx = totalDx / pathMag;
        const normDz = totalDz / pathMag;
        const world = relToWorld(normDx, normDz, playerYaw);
        const wdx = Math.sign(world.dx);
        const wdz = Math.sign(world.dz);

        const idx = terrainIdx(wdx, wdz);
        if (idx >= 0 && idx < 2601) {
            const cell = terrain[idx];
            if (cell === 1) score -= 2.0;
            if (cell === 2) score -= 5.0;
            if (cell === 3) score -= 3.0;
        }

        const first = primitive.segments[0];
        if (first) {
            const fWorld = relToWorld(Math.sign(first.dx), Math.sign(first.dz), playerYaw);
            const fIdx = terrainIdx(fWorld.dx, fWorld.dz);
            if (fIdx >= 0 && fIdx < 2601 && terrain[fIdx] === 2) {
                score -= 8.0;
            }
        }

        let passable = 0, blocked = 0;
        for (let gdz = -25; gdz <= 25; gdz++) {
            for (let gdx = -25; gdx <= 25; gdx++) {
                if (gdx === 0 && gdz === 0) continue;
                const dot = gdx * world.dx + gdz * world.dz;
                if (dot <= 0) continue;
                const gi = (gdz + 25) * 51 + (gdx + 25);
                if (gi < 0 || gi >= 2601) continue;
                if (terrain[gi] === 0) passable++;
                else blocked++;
            }
        }
        const total = passable + blocked;
        if (total > 0) {
            const openness = passable / total;
            if (openness < 0.4) score -= 3.0 * (1 - openness);
        }
    }

    if (primitive.name === 'backpedal') {
        const allMelee = threats.every(t => t.isMelee);
        if (allMelee && threats.length <= 2) score += 1.0;
        else score -= 1.0;
    }

    // Anchor/retreat constraints: avoid drifting too far from planner anchor.
    if (constraints?.returnRequired) {
        const maxR = Math.max(1, constraints.maxRetreatRadius || 24);
        const anchorDist = constraints._anchorDist || 0;
        const ax = constraints._anchorDx || 0;
        const az = constraints._anchorDz || 0;
        const amag = Math.sqrt(ax * ax + az * az);
        if (amag > 0) {
            const alignToAnchor = (totalDx * (ax / amag) + totalDz * (az / amag)) / pathMag; // +1 means toward anchor
            if (anchorDist > maxR * 0.8) score += alignToAnchor * 2.5;
            if (anchorDist > maxR) {
                score += alignToAnchor * 4.0;
                score -= Math.min((anchorDist - maxR) / maxR, 2) * 6.0;
            }
            // Hard veto: when way outside retreat radius, do not pick paths that move farther away.
            if (anchorDist > maxR * 1.3 && alignToAnchor < -0.2) return -1e9;
        }
    }

    return score;
}

/**
 * Check if this primitive immediately runs into a solid wall (code=1) within 2 blocks.
 * Used to hard-veto wall-colliding candidates (like entersLiquidSoon does for liquid).
 */
function hitsWallSoon(primitive, terrain, playerYaw) {
    if (!terrain || terrain.length < 2601 || !primitive?.segments?.length) return false;
    const seg = primitive.segments[0];
    const normMag = Math.sqrt(seg.dx * seg.dx + seg.dz * seg.dz) || 1;
    const ndx = seg.dx / normMag, ndz = seg.dz / normMag;
    const wd = relToWorld(ndx, ndz, playerYaw);
    // Check only 1-2 blocks ahead (not a wide corridor — walls need tighter check)
    for (let dist = 1; dist <= 2; dist++) {
        const idx = terrainIdx(wd.dx * dist, wd.dz * dist);
        if (idx >= 0 && idx < 2601 && terrain[idx] === 1) return true;
    }
    return false;
}

function entersLiquidSoon(primitive, terrain, playerYaw) {
    if (!terrain || terrain.length < 2601 || !primitive?.segments?.length) return false;
    // Check a wide corridor (not just 1 cell) along the path direction
    // Scan 1-4 blocks ahead in the path direction + 1 block to each side
    const seg = primitive.segments[0];
    const normMag = Math.sqrt(seg.dx * seg.dx + seg.dz * seg.dz) || 1;
    const ndx = seg.dx / normMag, ndz = seg.dz / normMag;
    // Convert to world space
    const wd = relToWorld(ndx, ndz, playerYaw);
    // Perpendicular direction for corridor width
    const perpX = -wd.dz, perpZ = wd.dx;
    for (let dist = 1; dist <= 4; dist++) {
        for (let side = -1; side <= 1; side++) {
            const checkX = wd.dx * dist + perpX * side;
            const checkZ = wd.dz * dist + perpZ * side;
            const idx = terrainIdx(checkX, checkZ);
            if (idx >= 0 && idx < 2601 && terrain[idx] === 2) return true;
        }
    }
    return false;
}

// ============================================================
// PATH EXECUTOR (world-space angle tracking, speed-aware dyaw)
// ============================================================
class PathExecutor {
    constructor() {
        this.currentPath = null;
        this.segmentIdx = 0;
        this.ticksInSegment = 0;
        this._stuckTicks = 0; // count consecutive ticks with near-zero speed
        this.commitUntil = 0;
    }

    setPath(primitive, playerYaw, commitMs = 800) {
        for (const seg of primitive.segments) {
            const relAngle = Math.atan2(seg.dx, seg.dz) * (180 / Math.PI);
            seg.worldAngle = playerYaw + relAngle;
        }
        this.currentPath = primitive;
        this.segmentIdx = 0;
        this.ticksInSegment = 0;
        this.commitUntil = Date.now() + commitMs;
    }

    shouldReplan() {
        if (!this.currentPath) return true;
        if (Date.now() >= this.commitUntil) return true;
        if (this.segmentIdx >= this.currentPath.segments.length) return true;
        return false;
    }

    isCommitted() {
        return this.currentPath && Date.now() < this.commitUntil;
    }

    tick(obs) {
        if (!this.currentPath || this.segmentIdx >= this.currentPath.segments.length) {
            return this._neutral();
        }

        const seg = this.currentPath.segments[this.segmentIdx];
        this.ticksInSegment++;

        if (this.ticksInSegment >= seg.ticks) {
            this.segmentIdx++;
            this.ticksInSegment = 0;
            if (this.segmentIdx >= this.currentPath.segments.length) {
                return this._neutral();
            }
        }

        const currentYaw = obs?.yaw || 0;
        const headingError = normalizeAngle(seg.worldAngle - currentYaw);
        const absError = Math.abs(headingError);

        // Emergency turn speed: MC has no turn speed limit. Old cap of 15°/tick
        // meant 12 ticks (0.6s) to turn 180° — bot ran 3+ blocks TOWARD mob during turn.
        // Now: 45°/tick when mob ahead (4 ticks for 180°), 30° for medium error, 15° when aligned.
        const speed = Math.sqrt((obs?.vx || 0) ** 2 + (obs?.vz || 0) ** 2);
        const maxDyaw = absError > 90 ? 45 : (absError > 45 ? 30 : 15);
        const gain = absError > 90 ? 0.9 : 0.6;
        const dyaw = Math.max(-maxDyaw, Math.min(maxDyaw, headingError * gain));

        let forward = false, back = false, left = false, right = false, sprint = false;

        if (absError > 120) {
            // Mob nearly directly ahead: do NOT forward (that runs into mob).
            // Strafe hard + turn fast. Strafe speed ~3.3 m/s vs forward 5.6 m/s,
            // but perpendicular to mob = net escape. Resumes forward once error < 120.
            forward = false;
            sprint = false;
            left = headingError < 0;
            right = headingError > 0;
        } else if (absError > 45) {
            // Mob at an angle: forward+sprint+strafe to arc away.
            forward = true;
            sprint = true;
            left = headingError < 0;
            right = headingError > 0;
        } else {
            forward = true;
            sprint = true;
        }

        // Jump when: wall directly ahead (step-up), OR stuck.
        // NOT when mob is close — jumping kills horizontal speed (airborne = ~0 forward),
        // making the bot SLOWER, letting mobs catch up. 69% of stuck frames were airborne
        // from unnecessary jumps. Only jump when terrain requires it.
        const terrain = obs?.terrain || [];
        let wallAhead = false;
        if ((forward || left || right) && terrain.length >= 2601) {
            const yaw = obs?.yaw || 0;
            const rad = yaw * Math.PI / 180;
            const fwdWx = -Math.sin(rad), fwdWz = Math.cos(rad);
            const idx1 = terrainIdx(fwdWx, fwdWz);
            if (idx1 >= 0 && idx1 < 2601 && terrain[idx1] === 1) wallAhead = true;
        }
        // Jump when wall ahead (step-up 1-block ledges). Without jump, bot gets stuck
        // at every terracotta step in badlands. MC auto-jump only works for players, not bots.
        const terrain = obs?.terrain || [];
        let wallAhead = false;
        if ((forward || left || right) && terrain.length >= 2601) {
            const yaw = obs?.yaw || 0;
            const rad = yaw * Math.PI / 180;
            const fwdWx = -Math.sin(rad), fwdWz = Math.cos(rad);
            const idx1 = terrainIdx(fwdWx, fwdWz);
            if (idx1 >= 0 && idx1 < 2601 && terrain[idx1] === 1) wallAhead = true;
        }
        let jump = !!obs?.onGround && wallAhead;

        // Stuck detection: escalating unstuck strategy.
        // ONLY count grounded stuck — airborne speed<0.03 is normal (falling/jumping)
        // and should NOT trigger stuck recovery. 69% of "stuck" was actually airborne.
        if (speed < 0.03 && (forward || back) && !!obs?.onGround) {
            this._stuckTicks++;
            if (this._stuckTicks >= 9) {
                // Phase 3: stuck 9+ ticks. If airborne, look down and place block to bridge.
                // If grounded, dig forward at feet level to break through.
                this.commitUntil = 0;
                this._stuckTicks = 0;
                if (!obs?.onGround && !obs?.inWater) {
                    // Airborne stuck: look straight down, place block (use key)
                    const pitchToDown = Math.max(-15, Math.min(15, (90 - (obs?.pitch || 0)) * 0.5));
                    return { forward: false, back: false, left: false, right: false,
                        sprint: false, jump: false, attack: false, use: true,
                        sneak: true, dyaw: 0, dpitch: pitchToDown };
                } else {
                    // Grounded stuck: look at wall ahead and dig (attack key)
                    const pitchToWall = Math.max(-15, Math.min(15, (0 - (obs?.pitch || 0)) * 0.3));
                    return { forward: true, back: false, left: false, right: false,
                        sprint: false, jump: false, attack: true,
                        dyaw: 0, dpitch: pitchToWall };
                }
            } else if (this._stuckTicks >= 6) {
                // Phase 2: stuck 6+ ticks. Dig wall ahead (hold attack + look straight)
                this.commitUntil = 0;
                const pitchToWall = Math.max(-15, Math.min(15, (0 - (obs?.pitch || 0)) * 0.3));
                return { forward: true, back: false, left: false, right: false,
                    sprint: false, jump: true, attack: true,
                    dyaw: 0, dpitch: pitchToWall };
            } else if (this._stuckTicks >= 3) {
                // Phase 1: stuck 3+ ticks. Jump + strafe to break free
                this.commitUntil = 0;
                const stuckDir = Math.random() > 0.5 ? 1 : -1;
                const stuckPitch = obs?.inWater ? Math.max(-15, Math.min(15, (-10 - (obs?.pitch || 0)) * 0.3)) : 0;
                return { forward: true, back: false, left: stuckDir < 0, right: stuckDir > 0,
                    sprint: true, jump: true, attack: false, dyaw: 12 * stuckDir, dpitch: stuckPitch };
            }
        } else {
            this._stuckTicks = 0;
        }

        // Always correct pitch toward horizontal — accumulated pitch drift (e.g. pitch=45°)
        // makes the bot look at the ground and miss attacks. Also helps sprint direction.
        let dpitch = 0;
        const currentPitch = obs?.pitch || 0;
        if (Math.abs(currentPitch) > 5) {
            dpitch = Math.max(-10, Math.min(10, -currentPitch * 0.3)); // gently pull toward 0
        }
        if (obs?.inWater) {
            if (back) {
                back = false;
                forward = true;
                sprint = true;
            }
            jump = true;
            // Correct pitch toward slightly above horizontal so forward moves horizontally
            const currentPitch = obs?.pitch || 0;
            const targetPitch = -10; // slightly upward helps surface
            dpitch = Math.max(-15, Math.min(15, (targetPitch - currentPitch) * 0.3));
        }

        // Counterattack: if mob is very close and cooldown ready, swing while fleeing.
        // This is the ONLY way bot can deal damage — fight mode never triggers (no weapon = lethal).
        // MC needs crosshair roughly on entity, so only attack when heading error is small.
        const nearestThreat = obs?.threats?.[0];
        const canCounterattack = nearestThreat
            && (nearestThreat.dist || 99) < 4
            && (obs?.attackCooldown || 0) >= 0.9
            && Math.abs(Math.atan2(nearestThreat.dx || 0, nearestThreat.dz || 0) * 180 / Math.PI) < 45;
        return { forward, back, left, right, sprint, jump, attack: !!canCounterattack, dyaw, dpitch };
    }

    _neutral() {
        return { forward: false, back: false, left: false, right: false, sprint: false, jump: false, attack: false, dyaw: 0, dpitch: 0 };
    }

    getInfo() {
        return {
            name: this.currentPath?.name || 'none',
            segment: this.segmentIdx,
            totalSegments: this.currentPath?.segments?.length || 0,
            committed: Date.now() < this.commitUntil,
        };
    }
}

// ============================================================
// LOCAL ESCAPE PLANNER (unified interface)
// ============================================================
export class LocalEscapePlanner {
    constructor() {
        this.threatField = new ThreatField();
        this.executor = new PathExecutor();
        this._lastPlanTime = 0;
        this._currentScore = -Infinity;
        this._logTick = 0;
        this._lastPressureTime = 0;
        this._lastEscapeYaw = null;
        this._lastLandPos = null; // { x, z } last known position on land
    }

    tick(obs, mode, constraints = null) {
        const terrain = obs?.terrain || [];
        const playerYaw = obs?.yaw || 0;
        let tf = this.threatField.compute(obs);

        // Track last known land position (for deep water fallback)
        if (!obs?.inWater && obs?.onGround && obs?.px !== undefined) {
            this._lastLandPos = { x: obs.px, z: obs.pz };
        }

        // In water: bias toward shore, or swim back to last land position
        if (obs?.inWater) {
            const shore = estimateShoreWorldDir(terrain);
            if (shore) {
                // Shore visible in terrain grid — blend shore direction with escape direction
                const esc = yawToWorldDir(tf.escapeYaw);
                const pressureNorm = Math.min(1, tf.pressure / 3);
                const shoreW = 0.8 - 0.4 * pressureNorm;
                const escW = 1 - shoreW;
                let wx = shore.x * shoreW + esc.x * escW;
                let wz = shore.z * shoreW + esc.z * escW;
                const mag = Math.sqrt(wx * wx + wz * wz) || 1;
                wx /= mag; wz /= mag;
                const fusedYaw = Math.atan2(-wx, wz) * (180 / Math.PI);
                const rel = worldToRel(wx, wz, playerYaw);
                tf = { ...tf, escapeYaw: fusedYaw, escapeDir: rel };
            } else if (this._lastLandPos && obs?.px !== undefined) {
                // Deep water: no shore visible. Swim toward last known land position.
                // IGNORE threat direction — getting to land is more important than fleeing.
                const dx = this._lastLandPos.x - obs.px;
                const dz = this._lastLandPos.z - obs.pz;
                const dist = Math.sqrt(dx * dx + dz * dz);
                if (dist > 1) {
                    const wx = dx / dist, wz = dz / dist;
                    const towardLandYaw = Math.atan2(-wx, wz) * (180 / Math.PI);
                    const rel = worldToRel(wx, wz, playerYaw);
                    tf = { ...tf, escapeYaw: towardLandYaw, escapeDir: rel };
                }
            }
            // else: no shore, no land memory — threat-based escape is the only option
        }

        if (tf.pressure > 0) {
            this._lastPressureTime = Date.now();
        }

        // --- Post-escape continuation ---
        if (tf.pressure === 0) {
            const msSincePressure = Date.now() - this._lastPressureTime;
            let continuationSafe = false;
            if (msSincePressure < 3000 && this._lastEscapeYaw !== null && terrain.length >= 2601) {
                const escRad = this._lastEscapeYaw * Math.PI / 180;
                const wdx = Math.sign(-Math.sin(escRad));
                const wdz = Math.sign(Math.cos(escRad));
                const idx = terrainIdx(wdx, wdz);
                const cell = (idx >= 0 && idx < 2601) ? terrain[idx] : 0;
                continuationSafe = (cell === 0);
            }
            if (msSincePressure < 3000 && this._lastEscapeYaw !== null && continuationSafe) {
                const headingError = normalizeAngle(this._lastEscapeYaw - playerYaw);
                const absError = Math.abs(headingError);
                const speed = Math.sqrt((obs?.vx || 0) ** 2 + (obs?.vz || 0) ** 2);
                const maxDyaw = absError > 90 ? 45 : (absError > 45 ? 30 : 15);
                const dyaw = Math.max(-maxDyaw, Math.min(maxDyaw, headingError * (absError > 90 ? 0.9 : 0.6)));
                const inWater = !!obs?.inWater;
                // Always correct pitch toward horizontal (not just in water)
                // Prevents pitch drift from baritone mining (pitch=45-72°) persisting in flee
                const currentPitch = obs?.pitch || 0;
                const contPitch = inWater
                    ? Math.max(-15, Math.min(15, (-10 - currentPitch) * 0.3))
                    : (Math.abs(currentPitch) > 5 ? Math.max(-10, Math.min(10, -currentPitch * 0.3)) : 0);
                return {
                    forward: true, // always forward+sprint, strafe to turn
                    back: false,
                    left: headingError < -30,
                    right: headingError > 30,
                    sprint: true,
                    jump: inWater,
                    attack: false,
                    dyaw,
                    dpitch: contPitch,
                };
            }
            this.executor.currentPath = null;
            this.executor.segmentIdx = 0;
            this.executor.ticksInSegment = 0;
            this.executor.commitUntil = 0;
            this._currentScore = -Infinity;
            this._lastEscapeYaw = null;
            // In water with no pressure: keep swimming toward shore, don't go neutral
            if (obs?.inWater) {
                const shore = estimateShoreWorldDir(terrain);
                let swimYaw = playerYaw;
                if (shore) {
                    swimYaw = Math.atan2(-shore.x, shore.z) * (180 / Math.PI);
                } else if (this._lastLandPos && obs?.px !== undefined) {
                    const dx = this._lastLandPos.x - obs.px;
                    const dz = this._lastLandPos.z - obs.pz;
                    const dist = Math.sqrt(dx * dx + dz * dz);
                    if (dist > 1) swimYaw = Math.atan2(-dx / dist, dz / dist) * (180 / Math.PI);
                }
                const he = normalizeAngle(swimYaw - playerYaw);
                const swimDyaw = Math.max(-10, Math.min(10, he * 0.4));
                const swimPitch = Math.max(-15, Math.min(15, (-10 - (obs?.pitch || 0)) * 0.3));
                return { forward: true, back: false, left: he < -30, right: he > 30,
                    sprint: true, jump: true, attack: false, dyaw: swimDyaw, dpitch: swimPitch };
            }
            return this.executor._neutral();
        }

        let mustReplan = this.executor.shouldReplan();
        const committed = this.executor.isCommitted();
        // Force replan if mob distance is decreasing during commit window
        // Prevents running toward mob with stale worldAngle after mob repositions
        const nearestDist = tf.threats.reduce((m, t) => Math.min(m, t.dist || 99), 99);
        if (committed && this._lastNearestDist !== undefined && nearestDist < this._lastNearestDist - 0.5) {
            mustReplan = true;
            this.executor.commitUntil = 0; // break commit
        }
        this._lastNearestDist = nearestDist;

        let candidates = generatePrimitives(tf.escapeDir, terrain);
        if (obs?.inWater) {
            candidates = candidates.filter(p => p.name !== 'backpedal');
        }
        // Fleeing from melee-only mobs: force straight-line escape, no strafing/zigzag
        // (strafe arcs circle back toward the mob)
        // Check nearest threat, not all — a far creeper shouldn't force strafe against a close zombie
        const nearest = tf.threats.reduce((c, t) => (!c || t.dist < c.dist) ? t : c, null);
        const nearestIsMelee = nearest && (nearest.isMelee || nearest.isExplosive); // creeper also needs straight-line
        if (nearestIsMelee) {
            candidates = candidates.filter(p =>
                p.name === 'sprint_escape' || p.name === 'retreat_straight' ||
                p.name === 'detour_left' || p.name === 'detour_right'
            );
        }
        // Hard veto on entering liquid when escaping on land.
        if (!obs?.inWater && terrain.length >= 2601) {
            candidates = candidates.filter(p => !entersLiquidSoon(p, terrain, playerYaw));
        }
        // Soft wall veto: prefer routes that don't immediately hit solid blocks.
        // Keep wall-hitting candidates as fallback if nothing else available.
        if (terrain.length >= 2601) {
            const noWall = candidates.filter(p => !hitsWallSoon(p, terrain, playerYaw));
            if (noWall.length > 0) candidates = noWall;
            // else: all routes have walls — keep all candidates, jump will handle 1-block steps
        }
        if (candidates.length === 0) {
            // Fallback: try all primitives but STILL filter liquid
            candidates = generatePrimitives(tf.escapeDir, terrain);
            if (!obs?.inWater && terrain.length >= 2601) {
                candidates = candidates.filter(p => !entersLiquidSoon(p, terrain, playerYaw));
            }
            // If still empty (all directions have water), pick shortest path away from threat
            if (candidates.length === 0) {
                candidates = generatePrimitives(tf.escapeDir, terrain).filter(p =>
                    p.name === 'backpedal' || p.name === 'retreat_straight'
                );
            }
        }
        let bestPrim = candidates[0];
        let bestScore = -Infinity;

        for (const prim of candidates) {
            let s = scorePath(prim, tf, terrain, playerYaw, constraints);
            if (obs?.inWater && prim.name === 'sprint_escape') s += 0.8;
            if (s > bestScore) {
                bestScore = s;
                bestPrim = prim;
            }
        }

        // Higher hysteresis: don't switch primitives easily during commit
        const hysteresis = committed ? 3.0 : 0.5;
        const shouldSwitch = mustReplan || bestScore > this._currentScore + hysteresis;
        if (shouldSwitch) {
            // Commit windows: short when wall blocked (need fast replan), longer when clear
            const wallNearby = terrain.length >= 2601 && [0,1,2,3,4,5,6,7].some(i => {
                const yaw = obs?.yaw || 0;
                const rad = (yaw + i * 45) * Math.PI / 180;
                const idx = terrainIdx(-Math.sin(rad), Math.cos(rad));
                return idx >= 0 && idx < 2601 && terrain[idx] === 1;
            });
            const commitMs = wallNearby ? 500 : (tf.pressure > 2 ? 900 : 1400);
            this.executor.setPath(bestPrim, playerYaw, commitMs);
            this._currentScore = bestScore;
        }

        // Store world-space escape yaw (stable, from ThreatField)
        this._lastEscapeYaw = tf.escapeYaw;

        const action = this.executor.tick(obs);

        this._logTick++;
        if (this._logTick % 10 === 0 && tf.pressure > 0) {
            const info = this.executor.getInfo();
            const nearest = tf.threats[0];
            const seg = this.executor.currentPath?.segments?.[this.executor.segmentIdx];
            const he = seg ? normalizeAngle(seg.worldAngle - playerYaw) : 0;
            const absHe = Math.abs(he);
            console.log(`[Escape] ${info.name} he=${he.toFixed(0)}° absHe=${absHe.toFixed(0)} fwd=${action.forward?1:0} spr=${action.sprint?1:0} L=${action.left?1:0} R=${action.right?1:0} dyaw=${action.dyaw.toFixed(1)} spd=${Math.sqrt((obs?.vx||0)**2+(obs?.vz||0)**2).toFixed(2)} | mob=${nearest?.type}@${nearest?.dist?.toFixed(1)} escYaw=${tf.escapeYaw.toFixed(0)} yaw=${playerYaw.toFixed(0)}`);
        }

        return action;
    }

    reset() {
        this.executor.currentPath = null;
        this.executor.segmentIdx = 0;
        this.executor.ticksInSegment = 0;
        this.executor.commitUntil = 0;
        this.executor._stuckTicks = 0;
        this._lastPlanTime = 0;
        this._currentScore = -Infinity;
        this._lastPressureTime = 0;
        this._lastEscapeYaw = null;
        this._lastLandPos = null;
        this.threatField._prevThreats = null;
    }

    getInfo() {
        return this.executor.getInfo();
    }
}
