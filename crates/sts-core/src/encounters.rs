//! `MonsterHelper.getEncounter`, with the same RNG consumption order.

use crate::monsters::beyond as be;
use crate::monsters::city as ci;
use crate::monsters::ending as en;
use crate::monsters::exordium as ex;
use crate::monsters::Monster;
use crate::rngs::Rngs;

pub const ACT1_WEAK: [&str; 4] = ["Cultist", "Jaw Worm", "2 Louse", "Small Slimes"];
pub const ACT1_STRONG: [&str; 10] = [
    "Blue Slaver",
    "Gremlin Gang",
    "Looter",
    "Large Slime",
    "Lots of Slimes",
    "Exordium Thugs",
    "Exordium Wildlife",
    "Red Slaver",
    "3 Louse",
    "2 Fungi Beasts",
];
pub const ACT1_ELITES: [&str; 3] = ["Gremlin Nob", "Lagavulin", "3 Sentries"];
pub const ACT1_BOSSES: [&str; 3] = ["The Guardian", "Hexaghost", "Slime Boss"];

pub fn all_act1() -> Vec<&'static str> {
    ACT1_WEAK.iter().chain(&ACT1_STRONG).chain(&ACT1_ELITES).chain(&ACT1_BOSSES).copied().collect()
}

pub const ACT2_WEAK: [&str; 5] = ["Spheric Guardian", "Chosen", "Shell Parasite", "3 Byrds", "2 Thieves"];
pub const ACT2_STRONG: [&str; 8] = [
    "Chosen and Byrds",
    "Sentry and Sphere",
    "Snake Plant",
    "Snecko",
    "Centurion and Healer",
    "Cultist and Chosen",
    "3 Cultists",
    "Shelled Parasite and Fungi",
];
pub const ACT2_ELITES: [&str; 3] = ["Gremlin Leader", "Slavers", "Book of Stabbing"];
pub const ACT2_BOSSES: [&str; 3] = ["Automaton", "Collector", "Champ"];
/// Fights only events start.
pub const ACT2_EVENT_FIGHTS: [&str; 3] = ["Masked Bandits", "Colosseum Slavers", "Colosseum Nobs"];

pub fn all_act2() -> Vec<&'static str> {
    ACT2_WEAK.iter().chain(&ACT2_STRONG).chain(&ACT2_ELITES).chain(&ACT2_BOSSES).chain(&ACT2_EVENT_FIGHTS).copied().collect()
}

pub const ACT3_WEAK: [&str; 3] = ["3 Darklings", "Orb Walker", "3 Shapes"];
pub const ACT3_STRONG: [&str; 8] = ["Spire Growth", "Transient", "4 Shapes", "Maw", "Sphere and 2 Shapes", "Jaw Worm Horde", "3 Darklings", "Writhing Mass"];
pub const ACT3_ELITES: [&str; 3] = ["Giant Head", "Nemesis", "Reptomancer"];
pub const ACT3_BOSSES: [&str; 3] = ["Awakened One", "Time Eater", "Donu and Deca"];
pub const ACT3_EVENT_FIGHTS: [&str; 2] = ["2 Orb Walkers", "Mysterious Sphere"];

pub fn all_act3() -> Vec<&'static str> {
    let mut v: Vec<&str> = ACT3_WEAK.to_vec();
    v.extend(ACT3_STRONG.iter().filter(|s| **s != "3 Darklings"));
    v.extend(ACT3_ELITES);
    v.extend(ACT3_BOSSES);
    v.extend(ACT3_EVENT_FIGHTS);
    v
}

/// Every encounter this port can build.
pub fn all() -> Vec<&'static str> {
    let mut v = all_act1();
    v.extend(all_act2());
    v.extend(all_act3());
    v.extend(["Shield and Spear", "The Heart"]);
    v
}

fn louse(r: &mut Rngs, asc: i32, x: f32) -> Monster {
    if r.misc.random_boolean() {
        ex::louse_normal(&mut r.monster_hp, asc, x)
    } else {
        ex::louse_defensive(&mut r.monster_hp, asc, x)
    }
}

fn slaver(r: &mut Rngs, asc: i32, x: f32) -> Monster {
    if r.misc.random_boolean() {
        ex::slaver_red(&mut r.monster_hp, asc, x)
    } else {
        ex::slaver_blue(&mut r.monster_hp, asc, x)
    }
}

/// `bottomGetWeakWildlife`: builds all three candidates, then picks one.
fn weak_wildlife(r: &mut Rngs, asc: i32, x: f32) -> Monster {
    let mut options = vec![louse(r, asc, x), ex::spike_slime_m(&mut r.monster_hp, asc, x), ex::acid_slime_m(&mut r.monster_hp, asc, x)];
    let k = r.misc.random_range(0, options.len() as i32 - 1) as usize;
    options.swap_remove(k)
}

fn strong_humanoid(r: &mut Rngs, asc: i32, x: f32) -> Monster {
    let mut options = vec![ex::cultist(&mut r.monster_hp, asc, x), slaver(r, asc, x), ex::looter(&mut r.monster_hp, asc, x)];
    let k = r.misc.random_range(0, options.len() as i32 - 1) as usize;
    options.swap_remove(k)
}

fn strong_wildlife(r: &mut Rngs, asc: i32, x: f32) -> Monster {
    let mut options = vec![ex::fungi_beast(&mut r.monster_hp, asc, x), ex::jaw_worm(&mut r.monster_hp, asc, x, false)];
    let k = r.misc.random_range(0, options.len() as i32 - 1) as usize;
    options.swap_remove(k)
}

pub(crate) fn gremlin(r: &mut Rngs, asc: i32, key: &str, x: f32) -> Monster {
    let rng = &mut r.monster_hp;
    match key {
        "GremlinWarrior" => ex::gremlin_warrior(rng, asc, x),
        "GremlinThief" => ex::gremlin_thief(rng, asc, x),
        "GremlinFat" => ex::gremlin_fat(rng, asc, x),
        "GremlinTsundere" => ex::gremlin_tsundere(rng, asc, x),
        _ => ex::gremlin_wizard(rng, asc, x),
    }
}

/// Builds the monster group for an encounter key, in group order.
pub fn create(key: &str, r: &mut Rngs, asc: i32) -> Result<Vec<Monster>, String> {
    let group = match key {
        "Blue Slaver" => vec![ex::slaver_blue(&mut r.monster_hp, asc, 0.0)],
        "Cultist" => vec![ex::cultist(&mut r.monster_hp, asc, 0.0)],
        "Jaw Worm" => vec![ex::jaw_worm(&mut r.monster_hp, asc, 0.0, false)],
        "Looter" => vec![ex::looter(&mut r.monster_hp, asc, 0.0)],
        "Gremlin Gang" => {
            let mut pool = vec![
                "GremlinWarrior",
                "GremlinWarrior",
                "GremlinThief",
                "GremlinThief",
                "GremlinFat",
                "GremlinFat",
                "GremlinTsundere",
                "GremlinWizard",
            ];
            let mut out = Vec::new();
            for x in [-320.0, -160.0, 25.0, 205.0] {
                let k = r.misc.random(pool.len() as i32 - 1) as usize;
                let key = pool.remove(k);
                out.push(gremlin(r, asc, key, x));
            }
            out
        }
        "Red Slaver" => vec![ex::slaver_red(&mut r.monster_hp, asc, 0.0)],
        "Large Slime" => {
            if r.misc.random_boolean() {
                vec![ex::acid_slime_l(&mut r.monster_hp, asc, 0.0)]
            } else {
                vec![ex::spike_slime_l(&mut r.monster_hp, asc, 0.0)]
            }
        }
        "Exordium Thugs" => {
            let a = weak_wildlife(r, asc, -160.0);
            let b = strong_humanoid(r, asc, 130.0);
            vec![a, b]
        }
        "Exordium Wildlife" => {
            let a = strong_wildlife(r, asc, -150.0);
            let b = weak_wildlife(r, asc, 150.0);
            vec![a, b]
        }
        "3 Louse" => {
            let a = louse(r, asc, -350.0);
            let b = louse(r, asc, -125.0);
            let c = louse(r, asc, 80.0);
            vec![a, b, c]
        }
        "2 Louse" => {
            let a = louse(r, asc, -200.0);
            let b = louse(r, asc, 80.0);
            vec![a, b]
        }
        "2 Fungi Beasts" => vec![ex::fungi_beast(&mut r.monster_hp, asc, -400.0), ex::fungi_beast(&mut r.monster_hp, asc, -40.0)],
        "Lots of Slimes" => {
            let mut pool = vec!["SpikeSlime_S", "SpikeSlime_S", "SpikeSlime_S", "AcidSlime_S", "AcidSlime_S"];
            let mut out = Vec::new();
            for x in [-480.0, -320.0, -160.0, 10.0, 200.0] {
                let k = r.misc.random(pool.len() as i32 - 1) as usize;
                let key = pool.remove(k);
                out.push(if key == "SpikeSlime_S" {
                    ex::spike_slime_s(&mut r.monster_hp, asc, x)
                } else {
                    ex::acid_slime_s(&mut r.monster_hp, asc, x)
                });
            }
            out
        }
        "Small Slimes" => {
            if r.misc.random_boolean() {
                let a = ex::spike_slime_s(&mut r.monster_hp, asc, -230.0);
                let b = ex::acid_slime_m(&mut r.monster_hp, asc, 35.0);
                vec![a, b]
            } else {
                let a = ex::acid_slime_s(&mut r.monster_hp, asc, -230.0);
                let b = ex::spike_slime_m(&mut r.monster_hp, asc, 35.0);
                vec![a, b]
            }
        }
        "Gremlin Nob" => vec![ex::gremlin_nob(&mut r.monster_hp, asc, 0.0, true)],
        "Lagavulin" => vec![ex::lagavulin(&mut r.monster_hp, asc, true)],
        "Lagavulin Event" => vec![ex::lagavulin(&mut r.monster_hp, asc, false)],
        "3 Sentries" => vec![
            ex::sentry(&mut r.monster_hp, asc, -330.0),
            ex::sentry(&mut r.monster_hp, asc, -85.0),
            ex::sentry(&mut r.monster_hp, asc, 140.0),
        ],
        "The Mushroom Lair" => vec![
            ex::fungi_beast(&mut r.monster_hp, asc, -450.0),
            ex::fungi_beast(&mut r.monster_hp, asc, -145.0),
            ex::fungi_beast(&mut r.monster_hp, asc, 180.0),
        ],
        "The Guardian" => vec![ex::the_guardian(&mut r.monster_hp, asc)],
        "Hexaghost" => vec![ex::hexaghost(&mut r.monster_hp, asc)],
        "Slime Boss" => vec![ex::slime_boss(&mut r.monster_hp, asc)],
        _ => return create_city(key, r, asc),
    };
    Ok(group)
}

fn create_city(key: &str, r: &mut Rngs, asc: i32) -> Result<Vec<Monster>, String> {
    let hp = &mut r.monster_hp;
    let group = match key {
        "2 Thieves" => {
            let a = ex::looter(hp, asc, -200.0);
            let b = ci::mugger(hp, asc, 80.0);
            vec![a, b]
        }
        "3 Byrds" => vec![ci::byrd(hp, asc, -360.0), ci::byrd(hp, asc, -80.0), ci::byrd(hp, asc, 200.0)],
        "4 Byrds" => vec![ci::byrd(hp, asc, -470.0), ci::byrd(hp, asc, -210.0), ci::byrd(hp, asc, 50.0), ci::byrd(hp, asc, 310.0)],
        "Chosen" => vec![ci::chosen(hp, asc, 0.0)],
        "Shell Parasite" => vec![ci::shelled_parasite(hp, asc, -20.0)],
        "Spheric Guardian" => vec![ci::spheric_guardian(asc, 0.0)],
        "Cultist and Chosen" => {
            let a = ex::cultist(hp, asc, -230.0);
            let b = ci::chosen(hp, asc, 100.0);
            vec![a, b]
        }
        "3 Cultists" => vec![ex::cultist(hp, asc, -465.0), ex::cultist(hp, asc, -130.0), ex::cultist(hp, asc, 200.0)],
        "Chosen and Byrds" => {
            let a = ci::byrd(hp, asc, -170.0);
            let b = ci::chosen(hp, asc, 80.0);
            vec![a, b]
        }
        "Sentry and Sphere" => {
            let a = ex::sentry(hp, asc, -305.0);
            vec![a, ci::spheric_guardian(asc, 0.0)]
        }
        "Snake Plant" => vec![ci::snake_plant(hp, asc, -30.0)],
        "Snecko" => vec![ci::snecko(hp, asc, 0.0)],
        "Centurion and Healer" => {
            let a = ci::centurion(hp, asc, -200.0);
            let b = ci::healer(hp, asc, 120.0);
            vec![a, b]
        }
        "Shelled Parasite and Fungi" => {
            let a = ci::shelled_parasite(hp, asc, -260.0);
            let b = ex::fungi_beast(hp, asc, 120.0);
            vec![a, b]
        }
        "Book of Stabbing" => vec![ci::book_of_stabbing(hp, asc)],
        "Gremlin Leader" => {
            // spawnGremlin: one random pick from the full pool (miscRng) for each.
            let pool = ["GremlinWarrior", "GremlinWarrior", "GremlinThief", "GremlinThief", "GremlinFat", "GremlinFat", "GremlinTsundere", "GremlinWizard"];
            let mut out = Vec::new();
            for x in [ci::GREMLIN_LEADER_X[0], ci::GREMLIN_LEADER_X[1]] {
                let k = r.misc.random_range(0, pool.len() as i32 - 1) as usize;
                out.push(gremlin(r, asc, pool[k], x));
            }
            out.push(ci::gremlin_leader(&mut r.monster_hp, asc));
            out
        }
        "Slavers" => {
            let a = ex::slaver_blue(hp, asc, -385.0);
            let b = ci::taskmaster(hp, asc, -133.0);
            let c = ex::slaver_red(hp, asc, 125.0);
            vec![a, b, c]
        }
        "Masked Bandits" => {
            let a = ci::bandit_pointy(hp, asc, -320.0);
            let b = ci::bandit_leader(hp, asc, -75.0);
            let c = ci::bandit_bear(hp, asc, 150.0);
            vec![a, b, c]
        }
        "Colosseum Nobs" => {
            let a = ci::taskmaster(hp, asc, -270.0);
            let b = ex::gremlin_nob(hp, asc, 130.0, true);
            vec![a, b]
        }
        "Colosseum Slavers" => {
            let a = ex::slaver_blue(hp, asc, -270.0);
            let b = ex::slaver_red(hp, asc, 130.0);
            vec![a, b]
        }
        "Automaton" => vec![ci::bronze_automaton(hp, asc)],
        "Champ" => vec![ci::champ(hp, asc)],
        "Collector" => vec![ci::the_collector(hp, asc)],
        _ => return create_beyond(key, r, asc),
    };
    Ok(group)
}

/// `getAncientShape`: a random shape (miscRng).
fn ancient_shape(r: &mut Rngs, asc: i32, x: f32) -> Monster {
    match r.misc.random(2) {
        0 => be::spiker(&mut r.monster_hp, asc, x),
        1 => be::repulsor(&mut r.monster_hp, asc, x),
        _ => be::exploder(&mut r.monster_hp, asc, x),
    }
}

/// `spawnShapes(weak)`: three or four shapes drawn from a pool of two of each.
fn shapes(r: &mut Rngs, asc: i32, count: usize) -> Vec<Monster> {
    let mut pool = vec!["Repulsor", "Repulsor", "Exploder", "Exploder", "Spiker", "Spiker"];
    let mut out = Vec::new();
    for x in [-480.0, -240.0, 0.0, 240.0].into_iter().take(count) {
        let k = r.misc.random(pool.len() as i32 - 1) as usize;
        let key = pool.remove(k);
        let hp = &mut r.monster_hp;
        out.push(match key {
            "Repulsor" => be::repulsor(hp, asc, x),
            "Spiker" => be::spiker(hp, asc, x),
            _ => be::exploder(hp, asc, x),
        });
    }
    out
}

fn create_beyond(key: &str, r: &mut Rngs, asc: i32) -> Result<Vec<Monster>, String> {
    let group = match key {
        "Reptomancer" => {
            let a = be::snake_dagger(&mut r.monster_hp, be::REPTOMANCER_X[1]);
            let b = be::reptomancer(&mut r.monster_hp, asc);
            let c = be::snake_dagger(&mut r.monster_hp, be::REPTOMANCER_X[0]);
            vec![a, b, c]
        }
        "Transient" => vec![be::transient(asc)],
        "3 Darklings" => {
            let hp = &mut r.monster_hp;
            vec![be::darkling(hp, asc, -440.0), be::darkling(hp, asc, -140.0), be::darkling(hp, asc, 180.0)]
        }
        "3 Shapes" => shapes(r, asc, 3),
        "4 Shapes" => shapes(r, asc, 4),
        "Jaw Worm Horde" => {
            let hp = &mut r.monster_hp;
            vec![ex::jaw_worm(hp, asc, -490.0, true), ex::jaw_worm(hp, asc, -150.0, true), ex::jaw_worm(hp, asc, 175.0, true)]
        }
        "Orb Walker" => vec![be::orb_walker(&mut r.monster_hp, asc, -30.0)],
        "2 Orb Walkers" => {
            let a = be::orb_walker(&mut r.monster_hp, asc, -250.0);
            let b = be::orb_walker(&mut r.monster_hp, asc, 150.0);
            vec![a, b]
        }
        "Spire Growth" => vec![be::spire_growth(&mut r.monster_hp, asc)],
        "Maw" => vec![be::maw(asc, -70.0)],
        "Sphere and 2 Shapes" => {
            let a = ancient_shape(r, asc, -435.0);
            let b = ancient_shape(r, asc, -210.0);
            vec![a, b, ci::spheric_guardian(asc, 110.0)]
        }
        "Mysterious Sphere" => {
            let a = ancient_shape(r, asc, -475.0);
            let b = ancient_shape(r, asc, -250.0);
            let c = be::orb_walker(&mut r.monster_hp, asc, 150.0);
            vec![a, b, c]
        }
        "Nemesis" => vec![be::nemesis(&mut r.monster_hp, asc)],
        "Writhing Mass" => vec![be::writhing_mass(&mut r.monster_hp, asc)],
        "Giant Head" => vec![be::giant_head(&mut r.monster_hp, asc)],
        "Time Eater" => vec![be::time_eater(&mut r.monster_hp, asc)],
        "Awakened One" => {
            let hp = &mut r.monster_hp;
            let a = ex::cultist(hp, asc, -590.0);
            let b = ex::cultist(hp, asc, -298.0);
            let c = be::awakened_one(hp, asc, 100.0);
            vec![a, b, c]
        }
        "Donu and Deca" => {
            let a = be::deca(&mut r.monster_hp, asc);
            let b = be::donu(&mut r.monster_hp, asc);
            vec![a, b]
        }
        "Shield and Spear" => {
            let a = en::spire_shield(&mut r.monster_hp, asc);
            let b = en::spire_spear(&mut r.monster_hp, asc);
            vec![a, b]
        }
        "The Heart" => vec![en::corrupt_heart(&mut r.monster_hp, asc)],
        _ => return Err(format!("unknown encounter '{key}'")),
    };
    Ok(group)
}
