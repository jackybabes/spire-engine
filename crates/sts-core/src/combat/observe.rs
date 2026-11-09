//! A JSON-friendly view of the combat for agents, and choice action enumeration.

use serde::{Deserialize, Serialize};

use super::{ChoiceKind, ChoicePile, Combat, Outcome, PendingChoice, PlayerAction};
use crate::cards::Card;
use crate::monsters::Intent;
use crate::powers::Power;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PowerView {
    pub id: String,
    pub amount: i32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CardView {
    pub id: String,
    pub name: String,
    pub upgraded: bool,
    pub cost: i32,
    pub card_type: String,
    pub damage: i32,
    pub block: i32,
    pub magic: i32,
    pub playable: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MonsterView {
    /// Stable id used as the `target` in actions.
    pub id: usize,
    pub name: String,
    pub hp: i32,
    pub max_hp: i32,
    pub block: i32,
    pub powers: Vec<PowerView>,
    pub intent: Intent,
    /// Damage per hit as displayed (after powers), or -1.
    pub intent_damage: i32,
    pub intent_hits: i32,
    pub alive: bool,
    /// Down but coming back (Darklings until all are down, the Awakened One between its forms):
    /// not alive, but still on the field, as the game shows it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub half_dead: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrbView {
    /// `Lightning`, `Frost`, `Dark`, `Plasma`, or `Empty` for an empty slot.
    pub id: String,
    pub passive: i32,
    pub evoke: i32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlayerView {
    pub hp: i32,
    pub max_hp: i32,
    pub block: i32,
    pub energy: i32,
    pub gold: i32,
    pub powers: Vec<PowerView>,
    /// Orb slots, front first (the Defect's; empty for the others).
    #[serde(default)]
    pub orbs: Vec<OrbView>,
    /// The stance (`Neutral`, `Calm`, `Wrath`, `Divinity`).
    #[serde(default)]
    pub stance: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChoiceView {
    pub kind: String,
    pub pile: ChoicePile,
    pub count: usize,
    pub any_number: bool,
    pub options: Vec<CardView>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Observation {
    pub turn: i32,
    pub outcome: Option<Outcome>,
    pub player: PlayerView,
    pub hand: Vec<CardView>,
    /// Draw pile contents sorted by name (order is hidden from the player in game).
    pub draw_pile: Vec<String>,
    pub discard_pile: Vec<String>,
    pub exhaust_pile: Vec<String>,
    pub monsters: Vec<MonsterView>,
    pub choice: Option<ChoiceView>,
    pub legal_actions: Vec<PlayerAction>,
}

fn powers(p: &[Power]) -> Vec<PowerView> {
    p.iter().map(|p| PowerView { id: p.java_id().to_string(), amount: p.amount }).collect()
}

fn sorted_names(cards: &[Card]) -> Vec<String> {
    let mut v: Vec<String> = cards.iter().map(|c| c.name()).collect();
    v.sort();
    v
}

impl Combat {
    fn card_view(&self, c: &Card, playable: bool) -> CardView {
        let mut tmp = c.clone();
        self.apply_powers_view(&mut tmp);
        // The game runs Strength and Dexterity through every card (a Strike's block becomes the
        // Dexterity, a Defend's damage the Strength) but shows damage only on attacks and block only
        // on cards that have block; so does the network's encoding (sts-py encode::hand_row).
        let shows_block = c.base_block > 0 || matches!(c.id, crate::cards::CardId::Stack | crate::cards::CardId::SpiritShield);
        CardView {
            id: c.data().java_id.to_string(),
            name: c.name(),
            upgraded: c.upgraded,
            // The cost the game shows (`AbstractCard.getCost`): 0 on a card free to play (an attack
            // under Swivel's Free Attack, a card played free once); X stays X.
            cost: if c.cost_for_turn >= 0 && self.free_to_play(c) { 0 } else { c.cost_for_turn },
            card_type: format!("{:?}", c.card_type()),
            damage: if c.card_type() == crate::cards::CardType::Attack { tmp.damage } else { 0 },
            block: if shows_block { tmp.block } else { 0 },
            magic: c.magic,
            playable,
        }
    }

    pub fn observe(&self) -> Observation {
        let legal = self.legal_actions();
        let hand = self
            .hand
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let playable = legal.iter().any(|a| matches!(a, PlayerAction::PlayCard { hand_index, .. } if *hand_index == i));
                self.card_view(c, playable)
            })
            .collect();
        let monsters = self
            .order
            .iter()
            .map(|&i| {
                let m = &self.monsters[i];
                // Runic Dome hides every intent.
                let dome = self.has_relic(crate::relics::RelicId::RunicDome);
                let attack = m.intent.is_attack() && !dome;
                MonsterView {
                    id: i,
                    name: m.id.java_id().to_string(),
                    hp: m.hp,
                    max_hp: m.max_hp,
                    block: m.block,
                    powers: powers(&m.powers),
                    intent: if dome { crate::monsters::Intent::Unknown } else { m.intent },
                    intent_damage: if attack { self.monster_intent_damage(i, m.intent_base_dmg) } else { -1 },
                    intent_hits: if attack { if m.is_multi_dmg { m.intent_multi } else { 1 } } else { 0 },
                    alive: !m.is_dead_or_escaped(),
                    half_dead: m.half_dead && !m.is_dying && !m.is_escaping,
                }
            })
            .collect();
        let choice = self.pending.as_ref().map(|p| {
            let pile = match p.pile {
                ChoicePile::Hand => &self.hand,
                ChoicePile::Discard => &self.discard_pile,
                ChoicePile::Exhaust => &self.exhaust_pile,
                ChoicePile::Offer => &p.offer,
                ChoicePile::Draw => &self.draw_pile,
            };
            ChoiceView {
                kind: choice_name(&p.kind).to_string(),
                pile: p.pile,
                count: p.count,
                any_number: p.any_number,
                options: p.options.iter().map(|&i| self.card_view(&pile[i], true)).collect(),
            }
        });
        Observation {
            turn: self.turn,
            outcome: self.outcome,
            player: PlayerView {
                hp: self.player.hp,
                max_hp: self.player.max_hp,
                block: self.player.block,
                energy: self.energy,
                gold: self.player.gold,
                powers: powers(&self.player.powers),
                orbs: self
                    .player
                    .orbs
                    .iter()
                    .map(|o| match o {
                        Some(o) => OrbView { id: o.id.java_id().to_string(), passive: self.orb_passive(o), evoke: self.orb_evoke(o) },
                        None => OrbView { id: "Empty".to_string(), passive: 0, evoke: 0 },
                    })
                    .collect(),
                stance: self.player.stance.java_id().to_string(),
            },
            hand,
            draw_pile: sorted_names(&self.draw_pile),
            discard_pile: self.discard_pile.iter().map(|c| c.name()).collect(),
            exhaust_pile: self.exhaust_pile.iter().map(|c| c.name()).collect(),
            monsters,
            choice,
            legal_actions: legal,
        }
    }
}

fn choice_name(k: &ChoiceKind) -> &'static str {
    match k {
        ChoiceKind::Exhaust => "exhaust",
        ChoiceKind::Discard { .. } => "discard",
        ChoiceKind::PutOnDeck => "put_on_deck",
        ChoiceKind::Armaments { .. } => "upgrade",
        ChoiceKind::DualWield { .. } => "duplicate",
        ChoiceKind::DiscardToTopOfDeck => "discard_to_top_of_deck",
        ChoiceKind::Exhume { .. } => "exhume",
        ChoiceKind::GamblingChip => "gambling_chip",
        ChoiceKind::Discover => "discover",
        ChoiceKind::Discovery { .. } => "discovery",
        ChoiceKind::Codex => "codex",
        ChoiceKind::DiscardToHand { .. } => "discard_to_hand",
        ChoiceKind::FromDeckToHand => "draw_pile_to_hand",
        ChoiceKind::Forethought => "to_bottom_of_draw_pile",
        ChoiceKind::Nightmare { .. } => "nightmare",
        ChoiceKind::Setup => "to_top_of_draw_pile",
        ChoiceKind::Retain => "retain",
        ChoiceKind::Recycle => "recycle",
        ChoiceKind::DrawPileToHand => "draw_pile_to_hand_pick",
        ChoiceKind::Scry => "scry",
        ChoiceKind::ChooseOne => "choose_one",
        ChoiceKind::ForeignInfluence { .. } => "foreign_influence",
        ChoiceKind::Meditate => "meditate",
        ChoiceKind::Omniscience { .. } => "omniscience",
    }
}

/// Enumerates selections for a pending choice. Exact-count choices of one card list each card;
/// larger selections list every combination up to a small cap.
pub fn choice_actions(p: &PendingChoice) -> Vec<PlayerAction> {
    let n = p.options.len();
    let mut out = Vec::new();
    let sizes: Vec<usize> = if p.any_number { ((if p.can_pick_zero { 0 } else { 1 })..=p.count.min(n)).collect() } else { vec![p.count] };
    for k in sizes {
        combos(&p.options, k, 0, &mut Vec::new(), &mut out);
        if out.len() > 4096 {
            break;
        }
    }
    out
}

fn combos(opts: &[usize], k: usize, start: usize, cur: &mut Vec<usize>, out: &mut Vec<PlayerAction>) {
    if cur.len() == k {
        out.push(PlayerAction::Choose { indices: cur.clone() });
        return;
    }
    for i in start..opts.len() {
        cur.push(opts[i]);
        combos(opts, k, i + 1, cur, out);
        cur.pop();
    }
}

#[cfg(test)]
mod card_view_tests {
    use crate::character::Character;
    use crate::combat::{Combat, CombatConfig};
    use crate::powers::{Power, PowerId};

    /// Dexterity shows on a Defend's block, not on a Strike; Strength on a Strike's damage, not
    /// on a Defend.
    #[test]
    fn powers_show_only_on_the_numbers_a_card_has() {
        let mut c = Combat::new(&CombatConfig::new(Character::Silent, 1, "Jaw Worm")).unwrap();
        c.player.powers.push(Power::new(PowerId::Dexterity, 2));
        c.player.powers.push(Power::new(PowerId::Strength, 3));
        let o = c.observe();
        for card in o.hand.iter().chain(o.choice.iter().flat_map(|x| x.options.iter())) {
            match card.name.as_str() {
                "Strike" => assert_eq!((card.damage, card.block), (9, 0)),
                "Defend" => assert_eq!((card.damage, card.block), (0, 7)),
                _ => {}
            }
        }
        assert!(o.hand.iter().any(|x| x.name == "Strike" || x.name == "Defend"));
    }

    /// Under Free Attack (Swivel) the game shows attacks at 0 and the other cards at their cost.
    #[test]
    fn free_attacks_show_cost_zero() {
        let mut c = Combat::new(&CombatConfig::new(Character::Watcher, 1, "Jaw Worm")).unwrap();
        c.player.powers.push(Power::new(PowerId::FreeAttack, 1));
        let o = c.observe();
        for card in &o.hand {
            match card.name.as_str() {
                "Strike" => assert_eq!(card.cost, 0),
                "Eruption" => assert_eq!(card.cost, 0),
                "Defend" | "Vigilance" => assert!(card.cost > 0),
                _ => {}
            }
        }
        assert!(o.hand.iter().any(|x| x.name == "Strike" || x.name == "Eruption"));
    }
}
