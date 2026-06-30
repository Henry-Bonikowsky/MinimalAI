"""AUTO-GENERATED from dimensions.json -- DO NOT EDIT DIRECTLY."""

import math

# ============================================================
#  Action Space (matches Java ActionSpace.java)
# ============================================================

# Movement (0-6)
ACT_FORWARD        = 0
ACT_BACKWARD       = 1
ACT_LEFT           = 2   # strafe left
ACT_RIGHT          = 3   # strafe right
ACT_JUMP           = 4
ACT_SNEAK          = 5
ACT_SPRINT         = 6

# Combat (7-13)
ACT_ATTACK         = 7   # auto-derived from ENGAGE
ACT_BLOCK          = 8   # 1.8 sword block
ACT_EAT_GAP        = 9   # golden apple
ACT_THROW_POT      = 10   # splash health pot
ACT_THROW_PEARL    = 11   # ender pearl
ACT_SPRINT_RESET   = 12   # toggle sprint for KB reset
ACT_SWAP_WEAPON    = 13   # always masked

# Sigil abilities (14-25)
ACT_SIGIL_0        = 14
ACT_SIGIL_11       = 25

# Sigil constants
SIGIL_BRACE_CHARGE_REQ = 30
SIGIL_BRACE_DR = 0.8
SIGIL_BRACE_DURATION = 200
SIGIL_BRACE_COOLDOWN = 600
SIGIL_CLEO_DMG_AMP = 1.2
SIGIL_CLEO_DURATION = 100
SIGIL_CLEO_COOLDOWN = 180
SIGIL_SAND_DMG_AMP = 1.3
SIGIL_SAND_DURATION = 120
SIGIL_SAND_COOLDOWN = 140
SIGIL_GRACE_DR = 0.3
SIGIL_GRACE_DURATION = 200
SIGIL_GRACE_COOLDOWN = 180

# Camera intent (26-29)
ACT_ENGAGE         = 26  # face target + auto-attack when in reach
ACT_FACE_AWAY      = 27  # look opposite of target (kiting)
ACT_LOOK_DOWN      = 28  # look at feet (self-pot)
ACT_FACE_MOVE      = 29  # face movement direction
ACT_FACE_TARGET    = 26  # alias for backward compat

# Target selection (30-34)
ACT_TARGET_0       = 30
ACT_TARGET_4       = 34

NUM_ACTIONS        = 35

# ============================================================
#  Model-Controlled Bits
#  Only these bits are output by the neural net.
#  Everything else is forced, masked, or auto-derived.
# ============================================================

USED_BITS = [0, 2, 3, 4, 9, 10, 11, 14, 15, 16, 17, 26]
# FWD, LEFT, RIGHT, JUMP, EAT, POT, PEARL, SIGIL0-3, ENGAGE
#
# Forced:  SPRINT(6)=always on, SNEAK(5)=always off
# Auto:    ATTACK(7)=derived from ENGAGE + reach
#          BACKWARD(1)=injected by combo style after sprint-hit
# Masked:  BLOCK(8), SPRINT_RESET(12), SWAP(13),
#          SIGILS(18-25), CAMERA(27-29), TARGETS(30-34)

# ============================================================
#  Observation Space
# ============================================================

SELF_STATE_DIM     = 38
ENTITY_FEATURE_DIM = 24
MAX_ENTITIES       = 8
COMBAT_CTX_DIM     = 26
SIGIL_STATE_DIM    = 48   # 12 slots * 4 dims
ENV_STATE_DIM      = 8
OBS_DIM            = 320  # total flat obs

# self_state slot map (0-37):
#  0  health/20           1  maxHealth/20         2  absorption/20
#  3  armor/20            4  armorToughness/20    5-7  velocity xyz
#  8  yaw/180             9  pitch/90            10  onGround
# 11  isSprinting        12  isSneaking          13  isBlocking
# 14  isUsingItem        15  fallDistance/10      16-18  weapon one-hot
# 19  attackDamage/20    20  goldenApples/64     21  splashPots/64
# 22  enderPearls/16     23  totems              24  gappleCooldown *
# 25  eatingProgress *   26  potLockout *        27  pearlCooldown *
# 28  regenActive *      29  regenRemaining *    30  ownComboStreak *
# 31  enemyComboStreak * 32  (unused)            33  hurtTime/10
# 34-37  sigil combat state (Java only)
# (* = sim-only, Java uses 24-32 for status effects)

# entity_features slot map (per entity, 24 dims):
#  0  alliance        1-3  dx/dy/dz /30     4  dist/30
#  5  health/20        6  maxHealth/20       7  armor/20
#  8-10  velocity xyz  11  yaw/180          12  onGround
# 13  isSprinting     14  isBlocking        15  isUsingItem
# 16-18  weapon 1-hot  19  attackDmg/20     20  hurtTime/10
# 21  isCurrentTarget  22-23  (unused)

# ============================================================
#  Network Architecture
# ============================================================

ENTITY_EMBED_DIM   = 64
ATTENTION_HEADS    = 2
ATTENTION_LAYERS   = 2
HIDDEN_DIM         = 256
GRU_HIDDEN_DIM     = 128

# ============================================================
#  Consumable Constants (match server kit)
# ============================================================

# Golden apples
GAP_EAT_TICKS        = 36    # 1.8s eat time
GAP_COOLDOWN_TICKS   = 180   # 9s cooldown
GAP_ABSORPTION       = 4.0   # absorption HP gained
GAP_REGEN_TICKS      = 100   # regen II duration
GAP_REGEN_RATE       = 25    # 1 HP every 25 ticks
GAP_START_COUNT      = 64    # per fight

# Splash potions (Instant Health II)
POT_HEAL_AMOUNT      = 8.0
POT_SPLASH_RADIUS    = 4.0
POT_THROW_LOCKOUT    = 3     # ticks can't attack during throw
POT_START_COUNT      = 31

# Ender pearls
PEARL_COOLDOWN_TICKS = 200  # 10s cooldown
PEARL_SELF_DAMAGE    = 5.0
PEARL_START_COUNT    = 16

# ============================================================
#  Combat Constants
# ============================================================

MELEE_ANGLE_DEG      = 60
MELEE_ANGLE_COS     = math.cos(math.radians(MELEE_ANGLE_DEG))
DEFAULT_ATTACK_REACH = 3.0
I_FRAME_TICKS        = 10
