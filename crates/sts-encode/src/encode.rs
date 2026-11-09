//! Fixed-size arrays for a fight, and the fixed action slots the network picks from.
//!
//! Every game becomes one row of `INTS` integers (IDs into learned embeddings, 0 = empty) and one
//! row of `FLOATS` numbers (raw, unscaled), plus a mask over `ACTIONS` slots. IDs come from the
//! engine's enums (`id + 1`), so new characters only add IDs; `layout()` names every field.
//!
//! Action slots:
//! - `play(h, t)`: hand card `h` (0-9) at target slot `t` (0 = no target, 1-6 = monster slot);
//! - `potion(p, t)`: potion slot `p` (0-4), same targets;
//! - `end_turn`;
//! - `pick(k)`: during a card selection, toggle option `k` (0-63, in the order the observation
//!   lists the options);
//! - `confirm`: finish a selection that allows fewer cards than the maximum.

use sts_core::cards::{Card, CardId, CardType};
use sts_core::combat::{ChoiceKind, ChoicePile, Orb, PendingChoice};
use sts_core::monsters::{Monster, MonsterId};
use sts_core::powers::PowerId;
use sts_core::relics::RelicId;
use sts_core::{Combat, PlayerAction};

/// The version of the network's inputs: raise it whenever an encoding changes what a network
/// sees (a field added, moved or resized, a feature computed differently, the action slots). A
/// released network's weights file stores the version it was trained on (`export_release.py`),
/// and its loader refuses a file from another version.
///
/// 2 (2026-10-24): hits, every enemy, random enemies, X, free, ethereal and kept on each hand
/// card (Feel No Pain's and Dark Embrace's columns gone); each hand card's damage over a whole
/// play against each monster; scalars for attacks played and cards discarded this turn, orb slots
/// and an extra turn; the stance, the last card type played, the orbs in slot order and each
/// monster's last two moves.
///
/// 3 (2026-10-30): between fights, the deck network's lookup (`encode_run` "opt_lookup" and
/// "run_lookup", written by the training code, zero from the engine).
pub const ENCODING_VERSION: u32 = 3;

pub const MAX_HAND: usize = 10;
pub const MAX_MONSTERS: usize = 6;
pub const MAX_MONSTER_POWERS: usize = 10;
pub const MAX_PLAYER_POWERS: usize = 20;
pub const MAX_RELICS: usize = 32;
pub const MAX_POTIONS: usize = 5;
pub const MAX_PILE: usize = 80;
pub const MAX_OPTIONS: usize = 64;
/// Orb slots: the game's most (slots are added one at a time, each stopping at 10).
pub const MAX_ORBS: usize = 10;
/// A monster's moves shown: its last two turns'.
pub const MOVES_SEEN: usize = 2;
pub const TARGETS: usize = 1 + MAX_MONSTERS;

pub const HAND_FEATS: usize = 15;
const MONSTER_FEATS: usize = 8;
const OPTION_FEATS: usize = 3;
pub const SCALARS: usize = 20;

pub const PLAY: usize = 0;
pub const POTION: usize = PLAY + MAX_HAND * TARGETS;
pub const END_TURN: usize = POTION + MAX_POTIONS * TARGETS;
pub const PICK: usize = END_TURN + 1;
pub const CONFIRM: usize = PICK + MAX_OPTIONS;
pub const ACTIONS: usize = CONFIRM + 1;

/// (name, length) of each integer field, in row order.
pub const INT_FIELDS: &[(&str, usize)] = &[
    ("character", 1),
    ("hand_card", MAX_HAND),
    ("player_power", MAX_PLAYER_POWERS),
    ("monster", MAX_MONSTERS),
    ("monster_intent", MAX_MONSTERS),
    ("monster_power", MAX_MONSTERS * MAX_MONSTER_POWERS),
    ("relic", MAX_RELICS),
    ("potion", MAX_POTIONS),
    ("draw_pile", MAX_PILE),
    ("discard_pile", MAX_PILE),
    ("exhaust_pile", MAX_PILE),
    ("choice_kind", 1),
    ("choice_pile", 1),
    ("option_card", MAX_OPTIONS),
    // Neutral, Calm, Wrath or Divinity.
    ("stance", 1),
    // The type of the last card played this combat (Follow-Up, Crush Joints, Sash Whip,
    // Sanctity look at it); 0 before the first.
    ("last_played_type", 1),
    // The orb slots, front (the next one evoked) first: Empty or the orb's kind, 0 past the last
    // slot (`orb_id`).
    ("orb", MAX_ORBS),
    // Each monster's moves on its previous two turns, the last first (`moves_seen`).
    ("monster_moves", MAX_MONSTERS * MOVES_SEEN),
];

/// (name, length) of each float field, in row order.
pub const FLOAT_FIELDS: &[(&str, usize)] = &[
    // hp, max_hp, block, energy, energy per turn, turn, hand/draw/discard/exhaust sizes, gold,
    // ascension, cards played this turn, choice count, choice may pick any number, cards picked,
    // attacks played this turn, cards discarded this turn, orb slots, an extra turn coming (Vault).
    ("scalars", SCALARS),
    // cost this turn, upgrades, damage and block as the game shows them (Strength, Weak, Dexterity,
    // Frail, Body Slam's block...), magic number, playable, needs a target, exhausts when played
    // (Corruption's skills too), hits per play, hits every enemy, hits random enemies, X (the
    // energy an X-cost card would spend), free this time, ethereal, kept at the end of the turn
    // (`hand_row`).
    ("hand", MAX_HAND * HAND_FEATS),
    ("player_power_amount", MAX_PLAYER_POWERS),
    // hp, max hp, block, intent damage (per hit, with powers), hits, targetable, half dead,
    // (intents, their damage and hits are 0 under Runic Dome, which hides them)
    // escaping.
    ("monster", MAX_MONSTERS * MONSTER_FEATS),
    ("monster_power_amount", MAX_MONSTERS * MAX_MONSTER_POWERS),
    ("relic_counter", MAX_RELICS),
    ("draw_pile_upgrades", MAX_PILE),
    ("discard_pile_upgrades", MAX_PILE),
    ("exhaust_pile_upgrades", MAX_PILE),
    // upgrades, cost, already picked.
    ("option", MAX_OPTIONS * OPTION_FEATS),
    // Each hand card's damage against each monster slot (`calculateCardDamage`: the monster's
    // Vulnerable and the like count; 0 for a card that isn't an attack).
    ("hand_target_damage", MAX_HAND * MAX_MONSTERS),
    // The same over one whole play: times its hits there (`play_damage_against`).
    ("hand_target_total", MAX_HAND * MAX_MONSTERS),
    // Each orb slot's numbers as the game shows them: Focus applied, a Dark orb's grown evoke.
    ("orb_passive", MAX_ORBS),
    ("orb_evoke", MAX_ORBS),
];

pub fn int_len() -> usize {
    INT_FIELDS.iter().map(|f| f.1).sum()
}

pub fn float_len() -> usize {
    FLOAT_FIELDS.iter().map(|f| f.1).sum()
}

/// Places per monster in the move vocabulary: moves are numbered 1-8 per monster, and 99 (an
/// escape) takes the last place.
const MOVES_PER_MONSTER: i32 = 16;

/// The sizes of the fight's own vocabularies beyond cards, powers, monsters and the like
/// (`sts.layout()["vocab"]`).
pub fn vocab_sizes() -> Vec<(&'static str, usize)> {
    vec![
        ("stance", 5),
        ("card_type", CardType::Curse as usize + 2),
        ("orb", 6),
        ("move", (MonsterId::CorruptHeart as usize + 1) * MOVES_PER_MONSTER as usize + 1),
    ]
}

/// A monster's move as a vocabulary id: each monster's moves have their own (0 = none).
pub fn move_id(monster: MonsterId, mv: u8) -> i32 {
    1 + monster as i32 * MOVES_PER_MONSTER + (mv as i32).min(MOVES_PER_MONSTER - 1)
}

/// A monster's moves on its previous turns, the last first: its `move_history` before the move
/// it is about to make (the last entry), as CommunicationMod's `last_move_id` and
/// `second_last_move_id`. Seen as they happened, so Runic Dome doesn't hide them.
pub fn moves_seen(m: &Monster) -> [i32; MOVES_SEEN] {
    let h = &m.move_history;
    let mut out = [0; MOVES_SEEN];
    for (k, x) in out.iter_mut().enumerate() {
        if h.len() >= k + 2 {
            *x = move_id(m.id, h[h.len() - 2 - k]);
        }
    }
    out
}

/// An orb slot: 1 empty, 2-5 Lightning, Frost, Dark, Plasma.
pub fn orb_id(o: Option<&Orb>) -> i32 {
    o.map_or(1, |o| o.id as i32 + 2)
}

/// Whether a hand card would stay in hand at the end of the turn: Retain, or (unless Ethereal)
/// Runic Pyramid or Equilibrium.
pub fn kept(c: &Combat, card: &Card) -> bool {
    card.retain || card.self_retain || (!card.ethereal && (c.has_relic(RelicId::RunicPyramid) || c.player.has_power(PowerId::Equilibrium)))
}

/// What one play of `card` does to monster `m`: a hit's damage times its hits there (`Combat::
/// card_hits`; random hits spread evenly over the `targets` monsters they can land on), or for
/// Judgement, a skill, the monster's HP and block when it would kill it.
pub fn play_damage_against(c: &Combat, card: &Card, m: usize, targets: usize) -> f32 {
    if card.id == CardId::Judgement {
        let x = &c.monsters[m];
        return if x.hp <= card.magic { (x.hp + x.block) as f32 } else { 0.0 };
    }
    let hits = c.card_hits(card, Some(m));
    if hits.count == 0 {
        return 0.0;
    }
    let total = c.card_damage_against(card, m) as f32 * hits.count as f32;
    match (hits.random, c.monsters[m].is_targetable()) {
        (false, _) => total,
        (true, true) => total / targets.max(1) as f32,
        (true, false) => 0.0,
    }
}

pub const CHOICE_KINDS: [&str; 24] = [
    "Exhaust",
    "Discard",
    "PutOnDeck",
    "Armaments",
    "DualWield",
    "DiscardToTopOfDeck",
    "Exhume",
    "GamblingChip",
    "Discover",
    "Discovery",
    "Codex",
    "DiscardToHand",
    "FromDeckToHand",
    "Forethought",
    "Nightmare",
    "Setup",
    "Retain",
    "Recycle",
    "DrawPileToHand",
    "Scry",
    "ChooseOne",
    "ForeignInfluence",
    "Meditate",
    "Omniscience",
];

fn choice_kind_id(k: &ChoiceKind) -> i32 {
    let name = format!("{k:?}");
    let name = name.split([' ', '{', '(']).next().unwrap_or("");
    CHOICE_KINDS.iter().position(|n| *n == name).map_or(0, |i| i as i32 + 1)
}

fn pile_id(p: ChoicePile) -> i32 {
    match p {
        ChoicePile::Hand => 1,
        ChoicePile::Draw => 2,
        ChoicePile::Discard => 3,
        ChoicePile::Exhaust => 4,
        ChoicePile::Offer => 5,
    }
}

/// The monsters shown in the observation (and targeted by slots 1-6): those still in the fight,
/// in group order.
pub fn monster_slots(c: &Combat) -> Vec<usize> {
    c.order.iter().copied().filter(|&m| !c.monsters[m].is_dying && !c.monsters[m].is_escaping).take(MAX_MONSTERS).collect()
}

/// The cards a pending choice's options refer to.
pub fn option_cards<'a>(c: &'a Combat, p: &'a PendingChoice) -> Vec<&'a Card> {
    let pile: &Vec<Card> = match p.pile {
        ChoicePile::Hand => &c.hand,
        ChoicePile::Draw => &c.draw_pile,
        ChoicePile::Discard => &c.discard_pile,
        ChoicePile::Exhaust => &c.exhaust_pile,
        ChoicePile::Offer => &p.offer,
    };
    p.options.iter().filter_map(|&i| pile.get(i)).collect()
}

pub struct Writer<'a, T> {
    pub buf: &'a mut [T],
    pub at: usize,
}

impl<'a, T: Copy + Default> Writer<'a, T> {
    /// Writes `vals` into a field of `len` slots (extra values are dropped, the rest stays 0).
    pub fn field(&mut self, len: usize, vals: impl IntoIterator<Item = T>) {
        for (i, v) in vals.into_iter().take(len).enumerate() {
            self.buf[self.at + i] = v;
        }
        self.at += len;
    }
}

fn card_id(c: &Card) -> i32 {
    c.id as i32 + 1
}

/// A hand card's row (see `HAND_FEATS`): what it would do in fight `c`, as the game shows it.
/// `exhausts` adds to the card's own exhaust flags (the real game reports Exhaust it was given).
pub fn hand_row(c: &Combat, card: &Card, playable: bool, targeted: bool, exhausts: bool) -> [f32; HAND_FEATS] {
    // The game runs Strength and Dexterity through every card (a Defend's damage becomes the
    // Strength, a Strike's block the Dexterity); only attacks show damage, and only cards with
    // block show block.
    let (damage, block) = c.card_display(card);
    let damage = if card.card_type() == CardType::Attack { damage } else { 0 };
    let block = if card.base_block > 0 { block } else { 0 };
    let exhausts = exhausts || card.exhaust || card.exhaust_on_use_once || (c.player.has_power(PowerId::Corruption) && card.card_type() == CardType::Skill);
    let hits = c.card_hits(card, None);
    let flag = |b: bool| b as i32 as f32;
    [
        card.cost_for_turn as f32,
        card.times_upgraded as f32,
        damage as f32,
        block as f32,
        card.magic as f32,
        playable as i32 as f32,
        targeted as i32 as f32,
        exhausts as i32 as f32,
        hits.count as f32,
        flag(hits.all),
        flag(hits.random),
        c.x_energy(card) as f32,
        flag(c.free_to_play(card)),
        flag(card.ethereal),
        flag(kept(c, card)),
    ]
}

/// Encodes `c` into `ints`, `floats` and `mask` (all zeroed by the caller). `legal` is
/// `c.legal_actions()`; `picked` holds the options already toggled in a pending selection.
pub fn encode_fight(c: &Combat, legal: &[PlayerAction], picked: &[usize], ints: &mut [i32], floats: &mut [f32], mask: &mut [u8]) {
    let slots = monster_slots(c);
    let slot_of = |m: usize| slots.iter().position(|&s| s == m);

    let mut w = Writer { buf: ints, at: 0 };
    w.field(1, [c.character as i32 + 1]);
    w.field(MAX_HAND, c.hand.iter().map(card_id));
    w.field(MAX_PLAYER_POWERS, c.player.powers.iter().map(|p| p.id as i32 + 1));
    w.field(MAX_MONSTERS, slots.iter().map(|&m| c.monsters[m].id as i32 + 1));
    // Runic Dome hides every intent from the player, so from the network too.
    let dome = c.has_relic(sts_core::relics::RelicId::RunicDome);
    w.field(MAX_MONSTERS, slots.iter().map(|&m| if dome { 0 } else { c.monsters[m].intent as i32 + 1 }));
    let mut mp = vec![0i32; MAX_MONSTERS * MAX_MONSTER_POWERS];
    for (j, &m) in slots.iter().enumerate() {
        for (k, p) in c.monsters[m].powers.iter().take(MAX_MONSTER_POWERS).enumerate() {
            mp[j * MAX_MONSTER_POWERS + k] = p.id as i32 + 1;
        }
    }
    w.field(mp.len(), mp);
    w.field(MAX_RELICS, c.player.relics.iter().map(|r| r.id as i32 + 1));
    w.field(MAX_POTIONS, c.potions.iter().map(|p| p.map_or(0, |p| p as i32 + 1)));
    // The draw pile's order is hidden from a player, so it is listed sorted.
    let mut draw: Vec<&Card> = c.draw_pile.iter().collect();
    draw.sort_by_key(|x| (x.id as i32, x.times_upgraded));
    w.field(MAX_PILE, draw.iter().map(|x| card_id(x)));
    w.field(MAX_PILE, c.discard_pile.iter().map(card_id));
    w.field(MAX_PILE, c.exhaust_pile.iter().map(card_id));
    let options: Vec<&Card> = c.pending.as_ref().map(|p| option_cards(c, p)).unwrap_or_default();
    w.field(1, [c.pending.as_ref().map_or(0, |p| choice_kind_id(&p.kind))]);
    w.field(1, [c.pending.as_ref().map_or(0, |p| pile_id(p.pile))]);
    w.field(MAX_OPTIONS, options.iter().map(|x| card_id(x)));
    w.field(1, [c.player.stance as i32 + 1]);
    w.field(1, [c.played_types.last().map_or(0, |t| *t as i32 + 1)]);
    w.field(MAX_ORBS, c.player.orbs.iter().map(|o| orb_id(o.as_ref())));
    w.field(MAX_MONSTERS * MOVES_SEEN, slots.iter().flat_map(|&m| moves_seen(&c.monsters[m])));

    let mut f = Writer { buf: floats, at: 0 };
    let pend = c.pending.as_ref();
    f.field(
        SCALARS,
        [
            c.player.hp as f32,
            c.player.max_hp as f32,
            c.player.block as f32,
            c.energy as f32,
            c.player.energy_master as f32,
            c.turn as f32,
            c.hand.len() as f32,
            c.draw_pile.len() as f32,
            c.discard_pile.len() as f32,
            c.exhaust_pile.len() as f32,
            c.player.gold as f32,
            c.ascension as f32,
            c.player.cards_played_this_turn as f32,
            pend.map_or(0.0, |p| p.count as f32),
            pend.map_or(0.0, |p| p.any_number as i32 as f32),
            picked.len() as f32,
            c.attacks_played_this_turn as f32,
            c.total_discarded_this_turn as f32,
            c.player.orbs.len() as f32,
            c.skip_monster_turn as i32 as f32,
        ],
    );
    let playable: Vec<bool> = (0..c.hand.len()).map(|h| legal.iter().any(|a| matches!(a, PlayerAction::PlayCard { hand_index, .. } if *hand_index == h))).collect();
    let mut hand = vec![0f32; MAX_HAND * HAND_FEATS];
    for (h, card) in c.hand.iter().take(MAX_HAND).enumerate() {
        let targeted = matches!(card.target(), sts_core::cards::CardTarget::Enemy | sts_core::cards::CardTarget::SelfAndEnemy);
        hand[h * HAND_FEATS..(h + 1) * HAND_FEATS].copy_from_slice(&hand_row(c, card, playable[h], targeted, false));
    }
    f.field(hand.len(), hand);
    f.field(MAX_PLAYER_POWERS, c.player.powers.iter().map(|p| p.amount as f32));
    let mut mon = vec![0f32; MAX_MONSTERS * MONSTER_FEATS];
    let mut mon_pow = vec![0f32; MAX_MONSTERS * MAX_MONSTER_POWERS];
    for (j, &m) in slots.iter().enumerate() {
        let x: &Monster = &c.monsters[m];
        let attack = x.intent_base_dmg >= 0 && !dome;
        let hits = if attack { if x.is_multi_dmg { x.intent_multi.max(1) } else { 1 } } else { 0 };
        let row = [
            x.hp as f32,
            x.max_hp as f32,
            x.block as f32,
            if attack { x.intent_dmg as f32 } else { 0.0 },
            hits as f32,
            x.is_targetable() as i32 as f32,
            x.half_dead as i32 as f32,
            x.is_escaping as i32 as f32,
        ];
        mon[j * MONSTER_FEATS..(j + 1) * MONSTER_FEATS].copy_from_slice(&row);
        for (k, p) in x.powers.iter().take(MAX_MONSTER_POWERS).enumerate() {
            mon_pow[j * MAX_MONSTER_POWERS + k] = p.amount as f32;
        }
    }
    f.field(mon.len(), mon);
    f.field(mon_pow.len(), mon_pow);
    f.field(MAX_RELICS, c.player.relics.iter().map(|r| r.counter as f32));
    f.field(MAX_PILE, draw.iter().map(|x| x.times_upgraded as f32));
    f.field(MAX_PILE, c.discard_pile.iter().map(|x| x.times_upgraded as f32));
    f.field(MAX_PILE, c.exhaust_pile.iter().map(|x| x.times_upgraded as f32));
    let mut opt = vec![0f32; MAX_OPTIONS * OPTION_FEATS];
    if let Some(p) = pend {
        for (k, card) in options.iter().take(MAX_OPTIONS).enumerate() {
            let chosen = picked.contains(&p.options[k]);
            opt[k * OPTION_FEATS..(k + 1) * OPTION_FEATS].copy_from_slice(&[card.times_upgraded as f32, card.cost_for_turn as f32, chosen as i32 as f32]);
        }
    }
    f.field(opt.len(), opt);
    let mut against = vec![0f32; MAX_HAND * MAX_MONSTERS];
    let mut total = vec![0f32; MAX_HAND * MAX_MONSTERS];
    let targets = slots.iter().filter(|&&m| c.monsters[m].is_targetable()).count();
    for (h, card) in c.hand.iter().take(MAX_HAND).enumerate() {
        for (j, &m) in slots.iter().enumerate() {
            against[h * MAX_MONSTERS + j] = c.card_damage_against(card, m) as f32;
            total[h * MAX_MONSTERS + j] = play_damage_against(c, card, m, targets);
        }
    }
    f.field(against.len(), against);
    f.field(total.len(), total);
    f.field(MAX_ORBS, c.player.orbs.iter().map(|o| o.as_ref().map_or(0.0, |o| c.orb_passive(o) as f32)));
    f.field(MAX_ORBS, c.player.orbs.iter().map(|o| o.as_ref().map_or(0.0, |o| c.orb_evoke(o) as f32)));

    // Action mask.
    if let Some(p) = pend {
        let need = selection_size(p);
        for (k, &o) in p.options.iter().take(MAX_OPTIONS).enumerate() {
            if !picked.contains(&o) && picked.len() < need.1 {
                mask[PICK + k] = 1;
            }
        }
        if p.any_number && picked.len() >= need.0 && picked.len() <= need.1 {
            mask[CONFIRM] = 1;
        }
        if need.1 == 0 {
            mask[CONFIRM] = 1;
        }
        return;
    }
    for a in legal {
        if let Some(s) = slot_for(a, &slot_of) {
            mask[s] = 1;
        }
    }
}

/// (fewest, most) cards a selection takes.
pub fn selection_size(p: &PendingChoice) -> (usize, usize) {
    let n = p.options.len();
    if p.any_number {
        (if p.can_pick_zero { 0 } else { 1.min(n) }, p.count.min(n))
    } else {
        (p.count.min(n), p.count.min(n))
    }
}

/// The action slot of a (non-selection) legal action.
pub fn slot_for(a: &PlayerAction, slot_of: &dyn Fn(usize) -> Option<usize>) -> Option<usize> {
    match a {
        PlayerAction::PlayCard { hand_index, target } if *hand_index < MAX_HAND => {
            let t = match target {
                None => 0,
                Some(m) => 1 + slot_of(*m)?,
            };
            Some(PLAY + hand_index * TARGETS + t)
        }
        PlayerAction::UsePotion { slot, target } if *slot < MAX_POTIONS => {
            let t = match target {
                None => 0,
                Some(m) => 1 + slot_of(*m)?,
            };
            Some(POTION + slot * TARGETS + t)
        }
        PlayerAction::EndTurn => Some(END_TURN),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sts_core::cards::CardId::*;
    use sts_core::CombatConfig;

    /// The hand's float rows and the per-target damage after `setup` changes a Jaw Worm fight.
    fn hand_rows(setup: impl Fn(&mut Combat)) -> (Vec<[f32; HAND_FEATS]>, Vec<f32>) {
        let mut c = Combat::new(&CombatConfig::ironclad(3, "Jaw Worm")).unwrap();
        setup(&mut c);
        let legal = c.legal_actions();
        let (mut ints, mut floats, mut mask) = (vec![0; int_len()], vec![0f32; float_len()], vec![0u8; ACTIONS]);
        encode_fight(&c, &legal, &[], &mut ints, &mut floats, &mut mask);
        let at = |name: &str| FLOAT_FIELDS.iter().take_while(|f| f.0 != name).map(|f| f.1).sum::<usize>();
        let hand = (0..c.hand.len()).map(|h| {
            let mut row = [0f32; HAND_FEATS];
            row.copy_from_slice(&floats[at("hand") + h * HAND_FEATS..at("hand") + (h + 1) * HAND_FEATS]);
            row
        });
        let a = at("hand_target_damage");
        (hand.collect(), floats[a..a + MAX_HAND * MAX_MONSTERS].to_vec())
    }

    fn hand_of(c: &mut Combat, ids: &[sts_core::cards::CardId]) {
        c.hand = ids.iter().map(|&id| c.new_card(id)).collect();
    }

    /// The hand shows damage and block as the game does, the damage against each monster counts
    /// its Vulnerable, and Corruption's skills show that they exhaust.
    #[test]
    fn hand_cards_show_what_they_would_do() {
        use sts_core::cards::CardId::*;
        use sts_core::powers::{Power, PowerId};
        // Strike: 6; Weak 6 -> 4; Strength 2 -> 8; Vulnerable monster -> 9.
        let (h, t) = hand_rows(|c| hand_of(c, &[StrikeR]));
        assert_eq!((h[0][2], t[0]), (6.0, 6.0));
        let (h, _) = hand_rows(|c| {
            hand_of(c, &[StrikeR]);
            c.player.powers.push(Power::new(PowerId::Weakened, 1));
        });
        assert_eq!(h[0][2], 4.0);
        let (h, _) = hand_rows(|c| {
            hand_of(c, &[StrikeR]);
            c.player.powers.push(Power::new(PowerId::Strength, 2));
        });
        assert_eq!(h[0][2], 8.0);
        let (_, t) = hand_rows(|c| {
            hand_of(c, &[StrikeR]);
            let m = c.order[0];
            c.monsters[m].powers.push(Power::new(PowerId::Vulnerable, 2));
        });
        assert_eq!(t[0], 9.0);
        // Body Slam deals the player's block; Defend shows Dexterity.
        let (h, t) = hand_rows(|c| {
            hand_of(c, &[BodySlam, DefendR]);
            c.player.block = 11;
            c.player.powers.push(Power::new(PowerId::Dexterity, 2));
        });
        assert_eq!((h[0][2], t[0]), (11.0, 11.0));
        assert_eq!(h[1][3], 7.0);
        // Strength and Dexterity don't give a Defend damage or a Strike block.
        let (h, _) = hand_rows(|c| {
            hand_of(c, &[StrikeR, DefendR]);
            c.player.powers.push(Power::new(PowerId::Strength, 3));
            c.player.powers.push(Power::new(PowerId::Dexterity, 2));
        });
        assert_eq!((h[0][2], h[0][3]), (9.0, 0.0));
        assert_eq!((h[1][2], h[1][3]), (0.0, 7.0));
        // Corruption makes skills exhaust: Defend shows it, Strike doesn't.
        let (h, t) = hand_rows(|c| {
            hand_of(c, &[DefendR, StrikeR]);
            c.player.powers.push(Power::flag(PowerId::Corruption));
        });
        assert_eq!((h[0][7], h[1][7]), (1.0, 0.0));
        assert_eq!((t[0], t[MAX_MONSTERS]), (0.0, 6.0)); // Defend deals nothing, Strike 6
    }

    /// A fight against `encounter` with exactly `hand`, after `setup`.
    fn fight(ch: sts_core::character::Character, encounter: &str, hand: &[CardId], setup: impl Fn(&mut Combat)) -> Combat {
        let mut c = Combat::new(&CombatConfig::new(ch, 3, encounter)).unwrap();
        hand_of(&mut c, hand);
        setup(&mut c);
        c
    }

    /// The row's integers and floats for fight `c`.
    fn encoded(c: &Combat) -> (Vec<i32>, Vec<f32>) {
        let (mut ints, mut floats, mut mask) = (vec![0; int_len()], vec![0f32; float_len()], vec![0u8; ACTIONS]);
        encode_fight(c, &c.legal_actions(), &[], &mut ints, &mut floats, &mut mask);
        (ints, floats)
    }

    fn field<'a, T>(fields: &[(&str, usize)], row: &'a [T], name: &str) -> &'a [T] {
        let at = fields.iter().take_while(|f| f.0 != name).map(|f| f.1).sum::<usize>();
        let len = fields.iter().find(|f| f.0 == name).expect(name).1;
        &row[at..at + len]
    }

    fn hand_row_of(c: &Combat, h: usize) -> Vec<f32> {
        field(FLOAT_FIELDS, &encoded(c).1, "hand")[h * HAND_FEATS..(h + 1) * HAND_FEATS].to_vec()
    }

    /// Hits, every enemy, random enemies and X (columns 8-11), and one play's damage against each
    /// monster (`hand_target_total`).
    #[test]
    fn hand_cards_show_their_hits() {
        use sts_core::character::Character::*;
        use sts_core::powers::{Power, PowerId};
        let c = fight(Ironclad, "Small Slimes", &[TwinStrike, Cleave, SwordBoomerang, Whirlwind, DefendR], |c| c.energy = 3);
        let row = |h: usize| hand_row_of(&c, h)[8..12].to_vec();
        assert_eq!(row(0), [2.0, 0.0, 0.0, 0.0]);
        assert_eq!(row(1), [1.0, 1.0, 0.0, 0.0]);
        assert_eq!(row(2), [3.0, 0.0, 1.0, 0.0]);
        assert_eq!(row(3), [3.0, 1.0, 0.0, 3.0]);
        assert_eq!(row(4), [0.0; 4]);
        let (_, floats) = encoded(&c);
        let total = field(FLOAT_FIELDS, &floats, "hand_target_total");
        let one = field(FLOAT_FIELDS, &floats, "hand_target_damage");
        let (m0, m1) = (c.order[0], c.order[1]);
        let d = |h: usize, m: usize| c.card_damage_against(&c.hand[h], m) as f32;
        assert_eq!((one[0], total[0], total[1]), (d(0, m0), 2.0 * d(0, m0), 2.0 * d(0, m1)));
        // Sword Boomerang's three hits land on either slime: half each.
        assert_eq!((total[2 * MAX_MONSTERS], total[2 * MAX_MONSTERS + 1]), (1.5 * d(2, m0), 1.5 * d(2, m1)));
        // Whirlwind hits each slime three times; Defend deals nothing.
        assert_eq!(total[3 * MAX_MONSTERS], 3.0 * d(3, m0));
        assert_eq!(total[4 * MAX_MONSTERS], 0.0);

        // Bane hits a poisoned monster twice.
        let c = fight(Silent, "Jaw Worm", &[Bane], |c| {
            let m = c.order[0];
            c.monsters[m].powers.push(Power::new(PowerId::Poison, 3));
        });
        let total = field(FLOAT_FIELDS, &encoded(&c).1, "hand_target_total").to_vec();
        assert_eq!(total[0], 2.0 * c.card_damage_against(&c.hand[0], c.order[0]) as f32);

        // Judgement (30) kills a monster at 30 HP or less: its HP and block; else nothing.
        for (hp, want) in [(25, 25.0 + 5.0), (40, 0.0)] {
            let c = fight(Watcher, "Jaw Worm", &[Judgement], |c| {
                let m = c.order[0];
                c.monsters[m].hp = hp;
                c.monsters[m].block = 5;
            });
            assert_eq!(field(FLOAT_FIELDS, &encoded(&c).1, "hand_target_total")[0], want, "{hp} HP");
        }
    }

    /// Free this time (column 12), ethereal (13) and kept at the end of the turn (14).
    #[test]
    fn hand_cards_show_cost_and_end_of_turn() {
        use sts_core::character::Character::*;
        use sts_core::powers::{Power, PowerId};
        let flags = |c: &Combat, h: usize| hand_row_of(c, h)[12..15].to_vec();
        let c = fight(Ironclad, "Jaw Worm", &[StrikeR, DefendR, GhostlyArmor], |c| c.hand[1].free_to_play_once = true);
        assert_eq!(flags(&c, 0), [0.0, 0.0, 0.0]);
        assert_eq!(flags(&c, 1), [1.0, 0.0, 0.0]);
        assert_eq!(flags(&c, 2), [0.0, 1.0, 0.0]);
        // Swivel's Free Attack: the next attack is free.
        let c = fight(Watcher, "Jaw Worm", &[StrikeP, DefendP, FlyingSleeves], |c| c.player.powers.push(Power::new(PowerId::FreeAttack, 1)));
        assert_eq!((flags(&c, 0)[0], flags(&c, 1)[0]), (1.0, 0.0));
        // Flying Sleeves retains itself.
        assert_eq!(flags(&c, 2)[2], 1.0);
        // Runic Pyramid and Equilibrium keep the hand, but not an Ethereal card.
        let pyramid = |c: &mut Combat| c.player.relics.push(sts_core::relics::RelicState::new(RelicId::RunicPyramid));
        let equilibrium = |c: &mut Combat| c.player.powers.push(Power::new(PowerId::Equilibrium, 1));
        for keep in [&pyramid as &dyn Fn(&mut Combat), &equilibrium] {
            let c = fight(Ironclad, "Jaw Worm", &[StrikeR, GhostlyArmor], keep);
            assert_eq!((flags(&c, 0)[2], flags(&c, 1)[2]), (1.0, 0.0));
        }
    }

    /// The stance, the last card type played and the turn's scalars.
    #[test]
    fn stance_last_type_and_scalars() {
        use sts_core::character::Character::*;
        use sts_core::combat::Stance;
        for (stance, id) in [(Stance::Neutral, 1), (Stance::Calm, 2), (Stance::Wrath, 3), (Stance::Divinity, 4)] {
            let c = fight(Watcher, "Jaw Worm", &[StrikeP], |c| c.player.stance = stance);
            assert_eq!(field(INT_FIELDS, &encoded(&c).0, "stance"), [id]);
        }
        let mut c = fight(Ironclad, "Jaw Worm", &[StrikeR, DefendR], |_| {});
        let last = |c: &Combat| field(INT_FIELDS, &encoded(c).0, "last_played_type")[0];
        let scalars = |c: &Combat| field(FLOAT_FIELDS, &encoded(c).1, "scalars")[16..20].to_vec();
        assert_eq!((last(&c), scalars(&c)), (0, vec![0.0, 0.0, 0.0, 0.0]));
        let m = c.order[0];
        c.step(&PlayerAction::PlayCard { hand_index: 0, target: Some(m) }).unwrap();
        assert_eq!((last(&c), scalars(&c)), (1, vec![1.0, 0.0, 0.0, 0.0])); // an attack
        c.step(&PlayerAction::PlayCard { hand_index: 0, target: None }).unwrap();
        assert_eq!((last(&c), scalars(&c)), (2, vec![1.0, 0.0, 0.0, 0.0])); // a skill
        c.total_discarded_this_turn = 2;
        c.skip_monster_turn = true;
        assert_eq!(scalars(&c), [1.0, 2.0, 0.0, 1.0]);
    }

    /// The orbs in slot order, empty slots included, with the numbers the game shows.
    #[test]
    fn orbs_in_slot_order() {
        use sts_core::character::Character::*;
        use sts_core::combat::{Orb, OrbId};
        use sts_core::powers::{Power, PowerId};
        let c = fight(Defect, "Jaw Worm", &[StrikeB], |c| {
            let mut dark = Orb::new(OrbId::Dark);
            dark.dark_evoke = 14;
            c.player.orbs = vec![Some(Orb::new(OrbId::Frost)), Some(dark), Some(Orb::new(OrbId::Lightning)), None];
            c.player.powers.push(Power::new(PowerId::Focus, 2));
        });
        let (ints, floats) = encoded(&c);
        assert_eq!(field(INT_FIELDS, &ints, "orb"), [3, 4, 2, 1, 0, 0, 0, 0, 0, 0]);
        assert_eq!(&field(FLOAT_FIELDS, &floats, "orb_passive")[..5], [4.0, 8.0, 5.0, 0.0, 0.0]);
        assert_eq!(&field(FLOAT_FIELDS, &floats, "orb_evoke")[..5], [7.0, 14.0, 10.0, 0.0, 0.0]);
        assert_eq!(field(FLOAT_FIELDS, &floats, "scalars")[18], 4.0);
        // No slots, nothing shown.
        let c = fight(Ironclad, "Jaw Worm", &[StrikeR], |_| {});
        assert_eq!(field(INT_FIELDS, &encoded(&c).0, "orb"), [0; MAX_ORBS]);
    }

    /// Each monster's moves on its last two turns, the last first; Runic Dome doesn't hide them.
    #[test]
    fn monsters_last_two_moves() {
        use sts_core::character::Character::*;
        for dome in [false, true] {
            let mut c = fight(Ironclad, "Jaw Worm", &[DefendR], |c| {
                if dome {
                    c.player.relics.push(sts_core::relics::RelicState::new(RelicId::RunicDome));
                }
            });
            let m = c.order[0];
            let moves = |c: &Combat| field(INT_FIELDS, &encoded(c).0, "monster_moves")[..MOVES_SEEN].to_vec();
            assert_eq!(moves(&c), [0, 0], "the first turn has none");
            for turn in 1..=3 {
                c.step(&PlayerAction::EndTurn).unwrap();
                let h = c.monsters[m].move_history.clone();
                let want: Vec<i32> = (0..MOVES_SEEN).map(|k| if h.len() >= k + 2 { move_id(MonsterId::JawWorm, h[h.len() - 2 - k]) } else { 0 }).collect();
                assert_eq!(moves(&c), want, "turn {turn}");
                assert!(want[0] > 0 && (turn < 2 || want[1] > 0), "turn {turn}: {want:?}");
            }
        }
        // Moves are numbered per monster.
        assert_ne!(move_id(MonsterId::JawWorm, 1), move_id(MonsterId::Cultist, 1));
        assert!((move_id(MonsterId::CorruptHeart, 99) as usize) < vocab_sizes().iter().find(|v| v.0 == "move").unwrap().1);
    }

    fn monster_fields(relics: Vec<RelicId>) -> (Vec<i32>, Vec<f32>) {
        let mut cfg = CombatConfig::ironclad(7, "Jaw Worm");
        cfg.relics = relics;
        let c = Combat::new(&cfg).unwrap();
        let (mut ints, mut floats, mut mask) = (vec![0; int_len()], vec![0.0; float_len()], vec![0; ACTIONS]);
        encode_fight(&c, &c.legal_actions(), &[], &mut ints, &mut floats, &mut mask);
        let at = |fields: &[(&str, usize)], name: &str| fields.iter().take_while(|f| f.0 != name).map(|f| f.1).sum::<usize>();
        let i = at(INT_FIELDS, "monster_intent");
        let f = at(FLOAT_FIELDS, "monster");
        (ints[i..i + MAX_MONSTERS].to_vec(), floats[f..f + MONSTER_FEATS].to_vec())
    }

    #[test]
    fn runic_dome_hides_intents() {
        let (intents, jaw_worm) = monster_fields(vec![RelicId::BurningBlood]);
        assert!(intents[0] > 0 && jaw_worm[3] > 0.0, "Jaw Worm opens with an attack");
        let (intents, jaw_worm) = monster_fields(vec![RelicId::BurningBlood, RelicId::RunicDome]);
        assert_eq!(intents[0], 0);
        assert_eq!((jaw_worm[3], jaw_worm[4]), (0.0, 0.0));
        assert!(jaw_worm[0] > 0.0, "HP is still shown");
    }
}
