# Dark Age of Camelot Recreation in Pure Zig
## A Technical Deep Dive for Hobby Development

---

## Executive Summary

Dark Age of Camelot (2001) was a technical marvel for its time - a 3-realm MMORPG that handled thousands of concurrent players per server cluster using surprisingly modest resources. Recreating it in pure Zig offers a unique opportunity to leverage modern language features (comptime, manual memory control, zero-cost abstractions) while targeting the original's visual fidelity and dramatically improving performance.

This document covers the complete architecture: networking, server infrastructure, game systems, client rendering, and a proposed implementation roadmap.

---

## Part 1: Understanding the Original Architecture

### Server Infrastructure (What We're Recreating)

The original DAoC architecture (per Wikipedia and the Mythic postmortem):

- **6 servers per "world"** - designed for 20,000 concurrent players, throttled to ~4,000
- **Thin client design** - most game logic ran server-side
- **~10 kbit/s per player** - remarkably low bandwidth budget
- **Linux/Open Source backend** - they ran on commodity hardware with open-source software
- **Development cost**: ~$2.5M over 18 months with 25 developers

### What OpenDAoC Taught Us

The modern [OpenDAoC](https://github.com/OpenDAoC/OpenDAoC-Core) emulator (C#) provides critical insights:

1. **ECS rewrite was essential** - They completely rewrote DOLSharp with ECS architecture for scalability
2. **Patch-level targeting** - Focused on 1.65 era, but architecture supports any patch
3. **~12,700 commits** - Shows the scope of work involved

### Packet Structure (From Security Research)

From the 2003 security advisory and Eve of Darkness documentation:

```
TCP Login Packet Format:
+------+------+-------------+--------------+
| 0x1b | 0x1b | Payload Len | Payload Data |
+------+------+-------------+--------------+
| ESC  | ESC  |  2 bytes    |   Variable   |

Client Payload:
+---------+---------+----------------------+
| Opcode  |    ?    | Opcode-specific data |
+---------+---------+----------------------+
| 2 bytes | 2 bytes | Payload Len - 4      |

Server Payload:
+---------+----------------------+
| Opcode  | Opcode-specific data |
+---------+----------------------+
| 2 bytes | Payload len - 2      |
```

Key opcodes discovered:
- `0x0065` - Server sends encryption key (13 bytes)
- `0x012c` - Login info
- `0x012d` - Account info
- `0x012e` - Account info (short)
- `0x0130` - Billing info

**All integers are network byte order (big-endian).**

---

## Part 2: Zig Architecture Decisions

### Why Zig is Perfect for This

1. **Comptime for game data** - Spell tables, class stats, item databases can be embedded and validated at compile time
2. **Zero-cost abstractions** - No GC pause concerns during combat
3. **Manual memory management** - Pool allocators for entities, arena allocators for frames
4. **C interop** - Can leverage existing libraries where needed
5. **Cross-compilation** - Single codebase for Linux server + Windows/Mac/Linux client

### Project Structure

```
daoc-zig/
├── build.zig
├── src/
│   ├── common/           # Shared code between client/server
│   │   ├── ecs/         # Entity Component System
│   │   ├── net/         # Packet definitions, serialization
│   │   ├── game/        # Game rules, formulas, data
│   │   └── math/        # 3D math, collision
│   │
│   ├── server/
│   │   ├── main.zig
│   │   ├── world/       # Zone management, spawns
│   │   ├── combat/      # Damage calculation, styles
│   │   ├── ai/          # NPC behavior
│   │   ├── db/          # Database layer
│   │   └── net/         # Server networking (io_uring)
│   │
│   └── client/
│       ├── main.zig
│       ├── render/      # Your graphics library
│       ├── ui/          # Game interface
│       ├── audio/       # Sound system
│       └── net/         # Client networking
│
├── data/
│   ├── spells/          # Spell definitions (comptime loaded)
│   ├── classes/         # Class stats and abilities
│   ├── items/           # Item database
│   ├── zones/           # Zone geometry, spawn data
│   └── npc/             # NPC templates
│
└── tools/
    ├── asset_converter/ # Convert original DAoC assets
    └── zone_editor/     # Zone editing tools
```

---

## Part 3: Entity Component System Design

### Archetype-Based ECS for Zig

Zig's comptime enables a particularly elegant ECS design. Based on research from [ZCS](https://github.com/Games-by-Mason/ZCS) and [Mach Engine's ECS](https://devlog.hexops.com/2022/lets-build-ecs-part-1/):

```zig
// Component definitions
pub const Position = struct {
    x: f32,
    y: f32,
    z: f32,
    heading: u16,  // 0-4095, DAoC uses 12-bit headings
    zone_id: u16,
};

pub const Health = struct {
    current: i32,
    max: i32,
    regen_rate: f32,
};

pub const Combat = struct {
    weapon_skill: u16,
    armor_factor: u16,
    absorption: u8,  // Percentage
    damage_type: DamageType,
};

pub const Character = struct {
    name: [24]u8,
    realm: Realm,
    class: Class,
    level: u8,
    spec_points: [8]u8,  // Spec line levels
};

pub const NpcAi = struct {
    behavior: AiBehavior,
    aggro_range: u16,
    leash_range: u16,
    faction_id: u16,
};

// Entity archetypes (comptime generated)
pub const PlayerArchetype = Archetype(&.{
    Position, Health, Combat, Character,
    Inventory, Stats, Buffs, Group,
});

pub const NpcArchetype = Archetype(&.{
    Position, Health, Combat, NpcAi,
    LootTable, SpawnPoint,
});

pub const SpellEffectArchetype = Archetype(&.{
    Position, Velocity, Duration, AreaEffect,
});
```

### System Design

```zig
// Systems operate on component queries
pub fn combatSystem(world: *World, dt: f32) void {
    // Query all entities with Combat + Health + Position
    var iter = world.query(&.{ Combat, Health, Position });
    
    while (iter.next()) |entity| {
        const combat = entity.get(Combat);
        const health = entity.get(Health);
        const pos = entity.get(Position);
        
        // Process pending attacks for this entity
        processPendingAttacks(world, entity, combat, health, pos);
    }
}

pub fn movementSystem(world: *World, dt: f32) void {
    var iter = world.query(&.{ Position, Velocity });
    
    while (iter.next()) |entity| {
        var pos = entity.getMut(Position);
        const vel = entity.get(Velocity);
        
        pos.x += vel.dx * dt;
        pos.y += vel.dy * dt;
        pos.z += vel.dz * dt;
    }
}

pub fn aiSystem(world: *World, dt: f32) void {
    var iter = world.query(&.{ NpcAi, Position, Combat });
    
    while (iter.next()) |entity| {
        const ai = entity.get(NpcAi);
        const pos = entity.get(Position);
        
        // Find nearby enemies, update aggro, pathfind, etc.
        updateNpcBehavior(world, entity, ai, pos);
    }
}
```

---

## Part 4: Combat System Implementation

### DAoC's Damage Formula (Reverse Engineered)

From community research:

```zig
pub fn calculateMeleeDamage(
    attacker: *const CombatStats,
    defender: *const CombatStats,
    weapon: *const Weapon,
) DamageResult {
    // Base damage calculation
    // Damage = WeapDmg * Delay * Quality * Condition * (WeapSkill / Target AF)
    
    const base_dps = weapon.dps;
    const delay = weapon.speed;
    const quality = @as(f32, weapon.quality) / 100.0;
    const condition = @as(f32, weapon.condition) / 100.0;
    
    // Weapon skill from spec + level + buffs
    const weapon_skill = calculateWeaponSkill(attacker);
    
    // Effective armor factor
    // Effective AF = Listed AF * Quality * (1 + (absorb * quality))
    const effective_af = calculateEffectiveAF(defender);
    
    var damage = base_dps * delay * quality * condition;
    damage *= @as(f32, weapon_skill) / effective_af;
    
    // Damage type vs armor type modifiers
    const type_modifier = getDamageTypeModifier(
        weapon.damage_type,
        defender.armor_type,
    );
    damage *= type_modifier;
    
    // Variance (typically 25%)
    const variance = 0.75 + (random.float(f32) * 0.5);
    damage *= variance;
    
    return .{
        .amount = @intFromFloat(damage),
        .type = weapon.damage_type,
        .critical = checkCritical(attacker, defender),
    };
}
```

### Damage Type vs Armor Matrix

```zig
pub const DamageTypeModifiers = struct {
    // Rows: Armor type, Columns: Damage type (Crush, Slash, Thrust)
    pub const table = [_][3]f32{
        // Cloth:     no modifiers
        .{ 1.00, 1.00, 1.00 },
        // Leather:   Crush -9.3%, Slash +7%, Thrust -18.3%
        .{ 0.907, 1.07, 0.817 },
        // Studded:   Crush -9.3%, Slash +7%, Thrust -18.3%
        .{ 0.907, 1.07, 0.817 },
        // Chain:     Crush -9.1%, Slash -30%, Thrust +9.3%
        .{ 0.909, 0.70, 1.093 },
        // Plate:     Crush +13%, Slash 0%, Thrust -11.1%
        .{ 1.13, 1.00, 0.889 },
    };
    
    pub fn get(armor: ArmorType, damage: DamageType) f32 {
        return table[@intFromEnum(armor)][@intFromEnum(damage)];
    }
};
```

### Combat Resolution Order

Per official sources, the order is:
1. Evade check
2. Parry check
3. Block check
4. Guard check
5. Miss check
6. Bladeturn check
7. Apply damage

```zig
pub fn resolveMeleeAttack(
    world: *World,
    attacker: Entity,
    defender: Entity,
    style: ?*const Style,
) AttackResult {
    const att_combat = attacker.get(Combat);
    const def_combat = defender.get(Combat);
    
    // 1. Evade
    if (checkEvade(def_combat)) return .{ .result = .evaded };
    
    // 2. Parry
    if (checkParry(def_combat, att_combat)) return .{ .result = .parried };
    
    // 3. Block (requires shield)
    if (def_combat.has_shield and checkBlock(def_combat)) {
        return .{ .result = .blocked };
    }
    
    // 4. Guard (from group member)
    if (checkGuard(world, defender)) return .{ .result = .guarded };
    
    // 5. Miss
    if (checkMiss(att_combat, def_combat)) return .{ .result = .missed };
    
    // 6. Bladeturn
    if (checkBladeturn(defender)) return .{ .result = .absorbed };
    
    // 7. Calculate and apply damage
    const damage = calculateMeleeDamage(att_combat, def_combat, att_combat.weapon);
    
    // Apply style bonuses/effects if used
    if (style) |s| {
        applyStyleEffects(world, attacker, defender, s, &damage);
    }
    
    return .{
        .result = .hit,
        .damage = damage,
    };
}
```

### Weapon Style System

DAoC's positional styles are a key feature:

```zig
pub const StyleCondition = enum {
    none,           // Always available
    after_evade,    // Must have just evaded
    after_parry,    // Must have just parried
    after_block,    // Must have just blocked
    from_behind,    // Must be behind target
    from_side,      // Must be beside target
    target_stunned, // Target must be stunned
    in_combat,      // Must be in active combat
    opening,        // No previous style used
};

pub const Style = struct {
    name: []const u8,
    spec_line: SpecLine,
    level_required: u8,
    endurance_cost: u16,
    
    condition: StyleCondition,
    followup_from: ?*const Style,  // Chain style
    
    damage_bonus: f32,      // Multiplier
    to_hit_bonus: i8,       // + to hit
    defense_bonus: i8,      // + to defense
    
    effect: ?StyleEffect,   // Stun, bleed, slow, etc.
    effect_chance: u8,      // Percentage
    
    growth_rate: f32,       // Damage scaling with spec
};

pub fn checkStyleCondition(
    attacker: Entity,
    defender: Entity,
    style: *const Style,
) bool {
    return switch (style.condition) {
        .none => true,
        .after_evade => attacker.get(CombatState).last_action == .evaded,
        .after_parry => attacker.get(CombatState).last_action == .parried,
        .from_behind => isPositionedBehind(attacker, defender),
        .from_side => isPositionedBeside(attacker, defender),
        // ...etc
    };
}
```

---

## Part 5: Networking Layer

### Server-Side: io_uring for Maximum Throughput

Using Mitchell Hashimoto's [libxev](https://github.com/mitchellh/libxev) or direct io_uring:

```zig
const std = @import("std");
const xev = @import("xev");

pub const GameServer = struct {
    loop: xev.Loop,
    listener: xev.TCP,
    clients: std.AutoHashMap(u32, *Client),
    world: *World,
    
    pub fn init(allocator: std.mem.Allocator, port: u16) !*GameServer {
        var self = try allocator.create(GameServer);
        
        self.loop = try xev.Loop.init(.{});
        self.listener = try xev.TCP.init(&self.loop, .{});
        self.clients = std.AutoHashMap(u32, *Client).init(allocator);
        
        const addr = try std.net.Address.parseIp4("0.0.0.0", port);
        try self.listener.bind(addr);
        try self.listener.listen(128);
        
        // Queue accept operation
        self.listener.accept(&self.loop, self, acceptCallback);
        
        return self;
    }
    
    pub fn run(self: *GameServer) !void {
        const tick_interval_ns = 50_000_000; // 50ms = 20 ticks/sec
        var last_tick = std.time.nanoTimestamp();
        
        while (true) {
            // Process IO events
            try self.loop.run(.{ .timeout_ns = tick_interval_ns });
            
            const now = std.time.nanoTimestamp();
            const dt = @as(f32, @floatFromInt(now - last_tick)) / 1_000_000_000.0;
            
            if (dt >= 0.05) {  // 50ms tick
                self.tick(dt);
                last_tick = now;
            }
        }
    }
    
    fn tick(self: *GameServer, dt: f32) void {
        // Run game systems
        combatSystem(self.world, dt);
        movementSystem(self.world, dt);
        aiSystem(self.world, dt);
        regenSystem(self.world, dt);
        buffSystem(self.world, dt);
        
        // Send state updates to clients
        self.broadcastWorldState();
    }
};
```

### Packet Serialization

Zig's comptime makes packet serialization elegant:

```zig
pub const PacketWriter = struct {
    buffer: []u8,
    pos: usize = 0,
    
    pub fn writeU16(self: *PacketWriter, value: u16) void {
        std.mem.writeInt(u16, self.buffer[self.pos..][0..2], value, .big);
        self.pos += 2;
    }
    
    pub fn writeU32(self: *PacketWriter, value: u32) void {
        std.mem.writeInt(u32, self.buffer[self.pos..][0..4], value, .big);
        self.pos += 4;
    }
    
    pub fn writeString(self: *PacketWriter, str: []const u8, max_len: usize) void {
        const len = @min(str.len, max_len);
        @memcpy(self.buffer[self.pos..][0..len], str[0..len]);
        self.pos += max_len;  // Fixed field width
    }
    
    pub fn finalize(self: *PacketWriter, opcode: u16) []u8 {
        // Insert header: ESC ESC LEN OPCODE
        const payload_len = self.pos;
        var header: [6]u8 = undefined;
        header[0] = 0x1b;
        header[1] = 0x1b;
        std.mem.writeInt(u16, header[2..4], @intCast(payload_len + 2), .big);
        std.mem.writeInt(u16, header[4..6], opcode, .big);
        
        // In practice, write header then payload to socket
        return self.buffer[0..self.pos];
    }
};

// Comptime packet definition
pub fn PacketDef(comptime fields: anytype) type {
    return struct {
        pub fn serialize(self: @This(), writer: *PacketWriter) void {
            inline for (fields) |field| {
                switch (@typeInfo(@TypeOf(@field(self, field.name)))) {
                    .Int => |info| {
                        if (info.bits == 16) writer.writeU16(@field(self, field.name))
                        else if (info.bits == 32) writer.writeU32(@field(self, field.name));
                    },
                    // ... handle other types
                }
            }
        }
    };
}
```

### Key Packet Types

```zig
pub const Opcodes = struct {
    // Login
    pub const CRYPT_KEY = 0x0065;
    pub const LOGIN_REQUEST = 0x012C;
    pub const LOGIN_RESPONSE = 0x00A8;
    pub const CHARACTER_LIST = 0x00FD;
    
    // World
    pub const PLAYER_POSITION = 0x00A1;
    pub const NPC_CREATE = 0x00DA;
    pub const PLAYER_CREATE = 0x0090;
    pub const OBJECT_UPDATE = 0x00D9;
    
    // Combat
    pub const COMBAT_ANIMATION = 0x0061;
    pub const DAMAGE_DEALT = 0x0073;
    pub const SPELL_CAST = 0x00B0;
    pub const SPELL_EFFECT = 0x00B4;
    
    // Social
    pub const CHAT_MESSAGE = 0x00AF;
    pub const GROUP_INVITE = 0x0060;
    pub const GUILD_INFO = 0x0093;
};

// Position update packet (sent 10x/sec from client)
pub const PlayerPositionPacket = struct {
    session_id: u16,
    sequence: u16,
    x: u32,        // Fixed point
    y: u32,
    z: u16,
    heading: u16,  // 0-4095
    flags: u8,     // Sitting, swimming, etc.
    speed: u8,
    
    pub fn parse(data: []const u8) PlayerPositionPacket {
        var reader = PacketReader{ .data = data };
        return .{
            .session_id = reader.readU16(),
            .sequence = reader.readU16(),
            .x = reader.readU32(),
            .y = reader.readU32(),
            .z = reader.readU16(),
            .heading = reader.readU16(),
            .flags = reader.readU8(),
            .speed = reader.readU8(),
        };
    }
};
```

---

## Part 6: Zone and World Management

### Zone Architecture

DAoC uses discrete zones with handoff between servers:

```zig
pub const Zone = struct {
    id: u16,
    name: []const u8,
    realm: Realm,
    
    // Bounds
    min_x: f32,
    max_x: f32,
    min_y: f32,
    max_y: f32,
    
    // Spatial partitioning (for entity queries)
    grid: SpatialGrid,
    
    // Zone-local entities
    npcs: std.ArrayList(Entity),
    objects: std.ArrayList(Entity),
    
    // Adjacent zones for handoff
    adjacent: [8]?u16,  // N, NE, E, SE, S, SW, W, NW
    
    pub fn getEntitiesInRange(
        self: *Zone,
        center: Position,
        radius: f32,
    ) []Entity {
        return self.grid.query(center.x, center.y, radius);
    }
};

pub const SpatialGrid = struct {
    cells: []std.ArrayList(Entity),
    cell_size: f32,
    width: u32,
    height: u32,
    
    pub fn getCellIndex(self: *SpatialGrid, x: f32, y: f32) usize {
        const cx = @intFromFloat((x - self.min_x) / self.cell_size);
        const cy = @intFromFloat((y - self.min_y) / self.cell_size);
        return cy * self.width + cx;
    }
    
    pub fn query(self: *SpatialGrid, x: f32, y: f32, radius: f32) []Entity {
        // Query all cells that could contain entities within radius
        const min_cx = @intFromFloat(@max(0, (x - radius - self.min_x) / self.cell_size));
        const max_cx = @intFromFloat(@min(self.width - 1, (x + radius - self.min_x) / self.cell_size));
        // ... similar for y
        
        var results = std.ArrayList(Entity).init(self.allocator);
        // Gather from relevant cells, filter by actual distance
        return results.toOwnedSlice();
    }
};
```

### NPC Spawning

```zig
pub const SpawnPoint = struct {
    x: f32,
    y: f32,
    z: f32,
    heading: u16,
    template_id: u32,
    respawn_time: u32,  // Seconds
    roam_radius: u16,
    
    // Runtime state
    current_entity: ?Entity = null,
    last_death_time: i64 = 0,
};

pub fn spawnSystem(world: *World, dt: f32) void {
    const now = std.time.timestamp();
    
    for (world.spawn_points.items) |*spawn| {
        if (spawn.current_entity == null) {
            if (now - spawn.last_death_time >= spawn.respawn_time) {
                const entity = createNpcFromTemplate(
                    world,
                    spawn.template_id,
                    spawn.x,
                    spawn.y,
                    spawn.z,
                );
                spawn.current_entity = entity;
            }
        }
    }
}
```

---

## Part 7: Client Architecture

### Integration with Your Graphics Library

Since you're building a Zig graphics library:

```zig
pub const GameClient = struct {
    // Your graphics lib
    renderer: *YourRenderer,
    
    // Game state (client-side prediction)
    world: *ClientWorld,
    local_player: ?Entity,
    
    // Network
    server_conn: *Connection,
    input_sequence: u32,
    pending_inputs: std.ArrayList(InputCommand),
    
    // UI
    ui_system: *UiSystem,
    
    pub fn frame(self: *GameClient, dt: f32) !void {
        // 1. Process server packets
        while (self.server_conn.recv()) |packet| {
            self.handleServerPacket(packet);
        }
        
        // 2. Handle input
        const input = self.gatherInput();
        if (input.hasMovement() or input.hasAction()) {
            self.pending_inputs.append(input);
            self.server_conn.send(input.serialize());
        }
        
        // 3. Client-side prediction
        self.predictMovement(dt);
        
        // 4. Interpolate other entities
        self.interpolateEntities(dt);
        
        // 5. Render
        self.render();
    }
    
    fn predictMovement(self: *GameClient, dt: f32) void {
        if (self.local_player) |player| {
            var pos = player.getMut(Position);
            const vel = player.get(Velocity);
            
            pos.x += vel.dx * dt;
            pos.y += vel.dy * dt;
            
            // Apply pending inputs that server hasn't acknowledged
            for (self.pending_inputs.items) |input| {
                if (input.sequence > self.last_acked_sequence) {
                    applyInput(pos, input);
                }
            }
        }
    }
    
    fn interpolateEntities(self: *GameClient, dt: f32) void {
        // Other players/NPCs use interpolation between server states
        for (self.world.getRemoteEntities()) |entity| {
            var pos = entity.getMut(Position);
            const interp = entity.get(Interpolation);
            
            const t = (self.render_time - interp.start_time) / interp.duration;
            pos.x = std.math.lerp(interp.start_x, interp.end_x, t);
            pos.y = std.math.lerp(interp.start_y, interp.end_y, t);
        }
    }
};
```

### Asset Loading (Original DAoC Format)

DAoC used custom formats you'll need to convert:

```zig
pub const AssetManager = struct {
    textures: std.StringHashMap(*Texture),
    models: std.StringHashMap(*Model),
    zones: std.StringHashMap(*ZoneGeometry),
    
    // DAoC used .mpk archives (modified ZIP)
    pub fn loadMpk(self: *AssetManager, path: []const u8) !void {
        var file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        
        // MPK is basically ZIP with custom header
        // Extract and parse contained files
    }
    
    // Original texture format: .dds (DirectDraw Surface)
    pub fn loadTexture(self: *AssetManager, name: []const u8) !*Texture {
        // DDSLoader implementation
    }
    
    // Original model format: .nif (NetImmerse/Gamebryo)
    pub fn loadModel(self: *AssetManager, name: []const u8) !*Model {
        // NIF loader - complex format, lots of documentation exists
    }
};
```

---

## Part 8: Database Design

### SQLite for Simplicity (Hobby Project)

```zig
const sqlite = @import("sqlite");

pub const Database = struct {
    conn: sqlite.Connection,
    
    pub fn loadCharacter(self: *Database, account_id: u32, slot: u8) !?Character {
        var stmt = try self.conn.prepare(
            \\SELECT name, realm, class, level, x, y, z, zone_id,
            \\       spec_points, inventory_blob, stats_blob
            \\FROM characters
            \\WHERE account_id = ? AND slot = ?
        );
        defer stmt.deinit();
        
        stmt.bind(.{ account_id, slot });
        
        if (try stmt.next()) |row| {
            return Character{
                .name = row.text(0),
                .realm = @enumFromInt(row.int(1)),
                .class = @enumFromInt(row.int(2)),
                .level = @intCast(row.int(3)),
                // ... etc
            };
        }
        return null;
    }
    
    pub fn saveCharacter(self: *Database, char: *const Character) !void {
        // Batch updates during server tick, not per-change
    }
};

// Schema
const schema = 
    \\CREATE TABLE accounts (
    \\    id INTEGER PRIMARY KEY,
    \\    username TEXT UNIQUE,
    \\    password_hash BLOB,
    \\    created_at INTEGER
    \\);
    \\
    \\CREATE TABLE characters (
    \\    id INTEGER PRIMARY KEY,
    \\    account_id INTEGER REFERENCES accounts(id),
    \\    slot INTEGER,
    \\    name TEXT,
    \\    realm INTEGER,
    \\    class INTEGER,
    \\    level INTEGER,
    \\    x REAL, y REAL, z REAL,
    \\    zone_id INTEGER,
    \\    spec_points BLOB,
    \\    inventory BLOB,
    \\    UNIQUE(account_id, slot)
    \\);
    \\
    \\CREATE TABLE items (
    \\    id INTEGER PRIMARY KEY,
    \\    template_id INTEGER,
    \\    owner_id INTEGER REFERENCES characters(id),
    \\    slot INTEGER,
    \\    durability INTEGER,
    \\    quality INTEGER
    \\);
;
```

---

## Part 9: Realm vs Realm (RvR) Systems

### Keep Siege

```zig
pub const Keep = struct {
    id: u16,
    name: []const u8,
    realm: Realm,
    
    // Structure
    walls: []WallSection,
    gates: []Gate,
    guards: []Entity,
    
    // Siege state
    under_siege: bool,
    attacking_realm: ?Realm,
    claim_guild: ?u32,
    
    // Upgrade level (affects guards, doors)
    level: u8,
    
    pub fn takeDamage(self: *Keep, section: *WallSection, damage: u32) void {
        section.health -= @min(section.health, damage);
        
        if (section.health == 0) {
            section.state = .breached;
            self.broadcastWallBreached(section);
        }
    }
    
    pub fn capture(self: *Keep, new_realm: Realm, claiming_guild: ?u32) void {
        self.realm = new_realm;
        self.claim_guild = claiming_guild;
        
        // Despawn old guards, spawn new ones
        for (self.guards) |guard| {
            guard.destroy();
        }
        self.spawnGuards(new_realm);
        
        // Update relic bonuses
        self.world.recalculateRealmBonuses();
    }
};

pub const SiegeWeapon = struct {
    type: SiegeType,
    health: u16,
    ammo: u8,
    operator: ?Entity,
    
    pub const SiegeType = enum {
        catapult,   // Anti-wall
        ballista,   // Anti-personnel
        ram,        // Anti-gate
        trebuchet,  // Long range anti-wall
    };
    
    pub fn fire(self: *SiegeWeapon, target: Position) !void {
        if (self.ammo == 0) return error.NoAmmo;
        if (self.operator == null) return error.NoOperator;
        
        self.ammo -= 1;
        
        // Create projectile entity
        const projectile = self.world.spawn(SiegeProjectileArchetype);
        projectile.set(Position, self.position);
        projectile.set(Velocity, calculateBallisticVelocity(self.position, target));
        projectile.set(SiegePayload, .{
            .damage = self.type.baseDamage(),
            .splash_radius = self.type.splashRadius(),
        });
    }
};
```

### Relic System

```zig
pub const Relic = struct {
    id: u8,
    type: RelicType,
    home_realm: Realm,
    current_realm: Realm,
    current_keep: ?u16,
    carrier: ?Entity,
    
    pub const RelicType = enum {
        strength,  // Melee damage bonus
        power,     // Magic damage bonus
    };
    
    pub fn getRealmBonus(self: *Relic) f32 {
        // Each captured relic gives realm-wide bonus
        if (self.current_realm != self.home_realm) {
            return switch (self.type) {
                .strength => 0.10,  // +10% melee damage
                .power => 0.10,     // +10% magic damage
            };
        }
        return 0;
    }
};
```

---

## Part 10: Implementation Roadmap

### Phase 1: Foundation (1-2 months)
- [ ] ECS implementation with basic components
- [ ] Packet serialization/deserialization
- [ ] Basic TCP networking (single server)
- [ ] SQLite database layer
- [ ] Login flow (account creation, character select)

### Phase 2: World (2-3 months)
- [ ] Zone loading and spatial partitioning
- [ ] Player movement and position sync
- [ ] NPC spawning and basic AI (stand, wander)
- [ ] Chat system (say, group, guild)

### Phase 3: Combat (2-3 months)
- [ ] Melee combat formulas
- [ ] Weapon styles and positional attacks
- [ ] Basic spell casting
- [ ] Health/mana/endurance regeneration
- [ ] Death and respawn

### Phase 4: Content (3-4 months)
- [ ] Complete class implementations (start with 3 base classes per realm)
- [ ] Spell system with all effect types
- [ ] Item system (equipping, stats, quality)
- [ ] NPC vendors and trainers

### Phase 5: RvR (2-3 months)
- [ ] Frontier zones
- [ ] Keep structures and siege
- [ ] Relic system
- [ ] Realm rank and abilities

### Phase 6: Polish (Ongoing)
- [ ] Client-side prediction improvements
- [ ] Performance optimization
- [ ] Bug fixing
- [ ] Content expansion

---

## Part 11: Zig Libraries to Leverage

### Recommended Dependencies

| Library | Purpose | URL |
|---------|---------|-----|
| libxev | Cross-platform event loop (io_uring/kqueue) | github.com/mitchellh/libxev |
| ZCS | Entity Component System | github.com/Games-by-Mason/ZCS |
| zig-network | TCP/UDP socket abstraction | github.com/ikskuh/zig-network |
| zig-sqlite | SQLite bindings | (various implementations) |
| zmath | SIMD math library | zig-gamedev |
| ztracy | Tracy profiler integration | zig-gamedev |

### Build Configuration

```zig
// build.zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    
    // Server executable
    const server = b.addExecutable(.{
        .name = "daoc-server",
        .root_source_file = b.path("src/server/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // Link io_uring on Linux
    if (target.result.os.tag == .linux) {
        server.linkLibC();
    }
    
    // Client executable
    const client = b.addExecutable(.{
        .name = "daoc-client",
        .root_source_file = b.path("src/client/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // Add your graphics library
    // client.addModule("renderer", your_renderer_module);
    
    b.installArtifact(server);
    b.installArtifact(client);
    
    // Test step
    const tests = b.addTest(.{
        .root_source_file = b.path("src/tests.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
}
```

---

## Part 12: Performance Targets

### Original DAoC Baseline
- 4,000 players per server cluster
- 10 kbit/s per player (~1.25 KB/s)
- 50ms server tick (20 ticks/second)

### Zig Recreation Targets (Conservative)
- **10,000+ players per server** (modern hardware, efficient code)
- **5 kbit/s per player** (better compression, delta encoding)
- **16ms server tick** (60 ticks/second for smoother combat)
- **<1ms frame time** for client (after render)

### Memory Budget
```
Per player:
  - Entity components: ~2KB
  - Inventory: ~4KB  
  - Pending packets: ~1KB
  - Total: ~8KB

10,000 players = 80MB base

Zone data:
  - Per zone: ~10MB (geometry, spawn data)
  - Active zones: 20 = 200MB

NPCs:
  - 50,000 NPCs × 1KB = 50MB

Total server memory: <500MB (extremely conservative)
```

---

## Conclusion

Recreating DAoC in pure Zig is ambitious but achievable as a hobby project. The key advantages:

1. **Zig's comptime** - Game data tables (spells, items, classes) can be validated at compile time
2. **Manual memory control** - No GC pauses during intense RvR combat
3. **Single codebase** - Shared logic between client and server
4. **Modern performance** - io_uring, SIMD math, cache-friendly ECS

The original game was built by 25 people in 18 months. A solo hobby recreation focusing on core gameplay (no expansions) is realistic over 1-2 years of weekend work.

**Recommended first milestone**: Get a character logged in, moving around a single zone, and killing a stationary NPC. Everything else builds from there.

---

## Resources

- [OpenDAoC Source](https://github.com/OpenDAoC/OpenDAoC-Core) - Modern C# emulator
- [Dawn of Light](https://github.com/Dawn-of-Light/DOLSharp) - Original emulator project
- [Mythic Postmortem](https://www.gamedeveloper.com/business/postmortem-mythic-entertainment-s-i-dark-age-of-camelot-i-) - Original development insights
- [Mach Engine ECS](https://devlog.hexops.com/2022/lets-build-ecs-part-1/) - Zig ECS tutorial
- [ZCS](https://github.com/Games-by-Mason/ZCS) - Production Zig ECS
- [Gaffer on Games](https://gafferongames.com/) - Game networking articles
- [disorder.dk/daoc](https://disorder.dk/daoc/) - Combat formula research
