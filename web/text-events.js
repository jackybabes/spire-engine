// What each event is and what its options do, in short hand-written words (no game text), as the
// game shows under each option: keyed by the engine's event id. `about`: a line under the title
// (a function of the event when it changes with the event's stage). `options`: keyed by how the
// engine's option label starts (the longest key that fits), each a function of the event's view
// `e` (its stage, rolled numbers `n`, and what it asks for), the run `o` (hp, max_hp, gold) and the
// ascension `a`, giving what the option does. The rules are the engine's
// (crates/sts-core/src/run/events*.rs); an option with no text here shows only its name.
const curse = (name) => `Become cursed: ${name}.`;
const relic = "Obtain a random relic.";

export const EVENT_TEXT = {
  // ---- Act 1 ----
  BigFish: {
    about: "A man by the road offers you a banana, a donut, or a box.",
    options: {
      banana: (e) => `Heal ${e.n[0]} HP.`,
      donut: () => "Gain 5 max HP.",
      box: () => `${relic} ${curse("Regret")}`,
    },
  },
  Cleric: {
    about: "A cleric offers healing and purification, for a price.",
    options: {
      "heal for": (e) => `Pay 35 gold. Heal ${e.n[0]} HP.`,
      purify: (e) => `Pay ${e.n[1]} gold. Remove a card from your deck.`,
    },
  },
  DeadAdventurer: {
    about: (e) => (e.stage === 1 ? "The elite that killed the adventurer wakes up!" : "A dead adventurer lies here. Whatever killed them may still be close."),
    options: {
      search: (e) => `Find gold, nothing or a relic (in a random order). ${e.n[0]}% chance the elite comes back instead (25% more after each search).`,
      fight: (e) => `Fight ${["3 Sentries", "Gremlin Nob", "Lagavulin"][e.n[1]] || "the elite"}. Win: ${e.n[2]} gold and what the body still holds.`,
    },
  },
  GoldenIdol: {
    about: (e) => (e.stage === 1 ? "A boulder rolls toward you!" : "A golden idol sits on a pedestal. It looks like a trap."),
    options: {
      "take the idol": () => "Obtain Golden Idol (25% more gold from fights). Then the trap goes off.",
      outrun: () => curse("Injury"),
      smash: (e) => `Take ${e.n[0]} damage.`,
      hide: (e) => `Lose ${e.n[1]} max HP.`,
    },
  },
  GoldenWing: {
    about: "A winged statue. Praying hurts; an Attack of 10 damage or more could break it open.",
    options: {
      pray: () => "Lose 7 HP. Remove a card from your deck.",
      destroy: () => "Gain 50-80 gold. Needs an Attack that deals 10 or more.",
    },
  },
  WorldOfGoop: {
    about: "You fall into a pool of goop. There is gold at the bottom.",
    options: {
      "gather gold": () => "Gain 75 gold. Lose 11 HP.",
      leave: (e) => (e.stage === 0 ? `Lose ${e.n[0]} gold.` : ""),
    },
  },
  LiarsGame: {
    about: "A serpent offers you gold, if you agree to something.",
    options: {
      agree: (e) => `Gain ${e.n[0]} gold. ${curse("Doubt")}`,
    },
  },
  LivingWall: {
    about: "A wall of faces lets you pass if you give it something.",
    options: {
      forget: () => "Remove a card from your deck.",
      change: () => "Transform a card.",
      grow: () => "Upgrade a card.",
    },
  },
  Mushrooms: {
    about: "A clearing full of mushrooms. Something moves among them.",
    options: {
      stomp: () => "Fight the mushrooms. Win: 20-30 gold and Odd Mushroom.",
      eat: (e, o) => `Heal ${Math.floor(o.max_hp * 0.25)} HP. ${curse("Parasite")}`,
    },
  },
  ScrapOoze: {
    about: "An ooze with something shiny inside it.",
    options: {
      "reach inside": (e) => `Lose ${e.n[0]} HP. ${e.n[1]}% chance to find a relic (10% more and 1 HP more each try).`,
    },
  },
  ShiningLight: {
    about: "A shining light. Stepping into it burns.",
    options: {
      enter: (e) => `Lose ${e.n[0]} HP. Upgrade 2 random cards.`,
    },
  },
  MatchAndKeep: {
    about: (e) => (e.stage === 2 ? `Turn two cards; a pair goes into your deck. Tries left: ${e.n[0]}.` : "A memory game: 12 cards face down, in pairs. Each pair you match goes into your deck."),
    options: {},
  },
  GoldenShrine: {
    about: "A golden shrine.",
    options: {
      pray: (e) => `Gain ${e.n[0]} gold.`,
      desecrate: () => `Gain 275 gold. ${curse("Regret")}`,
    },
  },
  Transmogrifier: { about: "A shrine that changes things.", options: { pray: () => "Transform a card." } },
  Purifier: { about: "A shrine that cleanses.", options: { pray: () => "Remove a card from your deck." } },
  UpgradeShrine: { about: "A shrine that strengthens.", options: { pray: () => "Upgrade a card." } },
  WheelOfChange: {
    about: (e) => (e.stage === 1 ? "The wheel stops." : "A wheel of fortune. Spin it and take what it lands on."),
    options: {
      spin: (e) => `One of: gold, a relic, a full heal, the curse Decay, a card removed, or losing ${e.n[1]}% of your max HP.`,
      gold: () => "Gold for this act (100, 200 or 300).",
      relic: () => relic,
      "full heal": () => "Heal to full HP.",
      curse: () => curse("Decay"),
      "remove a card": () => "Remove a card from your deck.",
      "lose hp": (e, o) => `Lose ${Math.floor(o.max_hp * e.n[1] / 100)} HP.`,
    },
  },
  AccursedBlacksmith: {
    about: "An abandoned forge. Something cursed lies in the rubble.",
    options: {
      forge: () => "Upgrade a card.",
      rummage: () => `Obtain Warped Tongs (upgrades a random card in your hand each turn). ${curse("Pain")}`,
    },
  },
  Bonfire: {
    about: "Spirits around a bonfire ask for an offering.",
    options: {
      approach: () => "Offer a card from your deck.",
      "offer a card": () => "It leaves your deck. A curse: Spirit Poop. Basic: nothing. Common: heal 5 HP. Uncommon: heal to full. Rare: gain 10 max HP and heal to full.",
    },
  },
  FountainOfCleansing: {
    about: "A fountain that washes curses away.",
    options: { drink: () => "Remove every curse from your deck (except Ascender's Bane, Curse of the Bell and Necronomicurse)." },
  },
  FaceTrader: {
    about: "A merchant of faces.",
    options: {
      touch: (e) => `Gain ${e.n[0]} gold. Take ${e.n[1]} damage.`,
      trade: () => "Trade your face for a random face relic.",
    },
  },
  Lab: { about: "An abandoned laboratory.", options: { search: (e, o, a) => `Find ${a >= 15 ? 2 : 3} random potions.` } },
  NoteForYourself: {
    about: "A note left by a past climber, with a card.",
    options: { "take the card": () => "Obtain the card. Then leave a card from your deck for a future climb (it leaves this deck)." },
  },
  WeMeetAgain: {
    about: "An old acquaintance wants something, and gives a relic for it.",
    options: {
      "give a potion": (e) => `Give ${e.potion || "a potion"}. ${relic}`,
      "give gold": (e) => `Give ${e.n[0]} gold. ${relic}`,
      "give a card": (e) => `Give ${e.give_card || "a card"}. ${relic}`,
      attack: () => "Nothing.",
    },
  },
  WomanInBlue: {
    about: "A woman sells potions.",
    options: {
      "20 gold": () => "Pay 20 gold. Obtain a random potion.",
      "30 gold": () => "Pay 30 gold. Obtain 2 random potions.",
      "40 gold": () => "Pay 40 gold. Obtain 3 random potions.",
      leave: (e, o, a) => (a >= 15 && e.stage === 0 ? `She hits you on the way out: lose ${Math.ceil(o.max_hp * 0.05)} HP.` : ""),
    },
  },

  // ---- Act 2 ----
  Addict: {
    about: "A desperate man has a relic he'll sell, or that you could take.",
    options: {
      "pay 85 gold": () => `Pay 85 gold. ${relic}`,
      rob: () => `${relic} ${curse("Shame")}`,
    },
  },
  BackToBasics: {
    about: "An ancient writing on simplicity.",
    options: {
      elegance: () => "Remove a card from your deck.",
      simplicity: () => "Upgrade every Strike and Defend.",
    },
  },
  Beggar: {
    about: "An old beggar asks for gold.",
    options: {
      "give 75 gold": () => "Pay 75 gold. Remove a card from your deck.",
      continue: () => "Remove a card from your deck.",
    },
  },
  Colosseum: {
    about: (e) => (e.stage === 2 ? "You won the first fight. The crowd wants more." : "You are thrown into an arena."),
    options: {
      fight: () => "Fight 2 Slavers.",
      cowardice: () => "Escape.",
      victory: () => "Fight 2 Gremlin Nobs. Win: a rare relic, an uncommon relic and 100 gold.",
    },
  },
  CursedTome: {
    about: "A cursed book. Each page costs more HP.",
    options: {
      read: () => "Lose 1, then 2, then 3 HP, then decide whether to take the book.",
      continue: (e) => `Lose ${Math.min(e.stage, 3)} HP.`,
      "take the book": (e) => `Lose ${e.n[0]} HP. Obtain a book: Necronomicon, Enchiridion or Nilry's Codex.`,
      "stop reading": () => "Lose 3 HP.",
    },
  },
  DrugDealer: {
    about: "A man offers experimental enhancements.",
    options: {
      "obtain j.a.x.": () => "Obtain J.A.X. (lose 3 HP, gain 2 Strength).",
      "transform 2 cards": () => "Transform 2 cards.",
      "mutagenic strength": () => "Obtain Mutagenic Strength (3 Strength at the start of each fight, lost at the end of your first turn).",
    },
  },
  ForgottenAltar: {
    about: "An altar that asks for a sacrifice.",
    options: {
      "offer golden idol": () => "Trade Golden Idol for Bloody Idol (heal 5 HP whenever you gain gold).",
      sacrifice: (e) => `Gain 5 max HP. Take ${e.n[0]} damage.`,
      desecrate: () => curse("Decay"),
    },
  },
  Ghosts: {
    about: "Ghosts offer to make you one of them.",
    options: {
      accept: (e, o, a) => `Lose ${e.n[0]} max HP. Obtain ${a >= 15 ? 3 : 5} Apparitions (become Intangible for a turn).`,
    },
  },
  MaskedBandits: {
    about: "Masked bandits demand your gold.",
    options: {
      "pay all gold": (e, o) => `Lose all your gold (${o.gold}).`,
      fight: () => "Fight the bandits. Win: 25-35 gold and Red Mask.",
    },
  },
  Nest: {
    about: (e) => (e.stage === 1 ? "A cult's donation box, and a line of cultists." : "You stumble into a cultist ceremony."),
    options: {
      "smash and grab": (e) => `Gain ${e.n[0]} gold.`,
      "stay in line": () => "Take 6 damage. Obtain Ritual Dagger.",
    },
  },
  TheLibrary: {
    about: (e) => (e.stage === 1 ? "Choose a book to take with you." : "A quiet library. Read, or sleep."),
    options: {
      read: () => "Choose 1 of 20 cards.",
      sleep: (e) => `Heal ${e.n[0]} HP.`,
      take: () => "Add this card to your deck.",
    },
  },
  TheMausoleum: {
    about: "A coffin in a mausoleum.",
    options: { "open the coffin": (e, o, a) => `${relic} ${a >= 15 ? curse("Writhe") : `50% chance: ${curse("Writhe").toLowerCase()}`}` },
  },
  Vampires: {
    about: "Vampires offer you their gift.",
    options: {
      accept: (e) => `Lose ${e.n[0]} max HP. Every Strike becomes one of 5 Bites.`,
      "give blood vial": () => "Lose Blood Vial. Every Strike becomes one of 5 Bites.",
    },
  },
  Designer: {
    about: "A fashion designer offers to tailor your deck.",
    options: {},
  },
  Duplicator: { about: "A shrine that copies.", options: { pray: () => "Duplicate a card in your deck." } },
  KnowingSkull: {
    about: "A skull that grants wishes for HP. Each wish costs 1 HP more than the last.",
    options: {
      "a potion": (e) => `Obtain a random potion. Lose ${e.n[0]} HP.`,
      "90 gold": (e) => `Gain 90 gold. Lose ${e.n[1]} HP.`,
      "a colorless card": (e) => `Obtain a random uncommon colorless card. Lose ${e.n[2]} HP.`,
      leave: (e) => (e.stage === 1 ? `Lose ${e.n[3]} HP.` : ""),
    },
  },
  Nloth: {
    about: "N'loth wants one of your relics.",
    options: { trade: () => "Give this relic. Obtain N'loth's Gift (rare cards appear in card rewards three times as often)." },
  },
  TheJoust: {
    about: "A joust between a murderer and the owner of a castle. Place a bet.",
    options: {
      "bet on the murderer": () => "Pay 50 gold. If the murderer wins (70%), gain 100 gold.",
      "bet on the owner": () => "Pay 50 gold. If the owner wins (30%), gain 250 gold.",
    },
  },

  // ---- Act 3 ----
  Falling: {
    about: (e) => (e.stage === 1 ? "You fall. Something has to go." : "The floor gives way."),
    options: { lose: () => "Remove this card from your deck." },
  },
  MindBloom: {
    about: "A strange voice offers you a choice.",
    options: {
      "i am war": (e, o, a) => `Fight a random Act 1 boss. Win: a rare relic and ${a >= 13 ? 25 : 50} gold.`,
      "i am awake": () => "Upgrade every card. Obtain Mark of the Bloom (you can no longer heal).",
      "i am rich": () => `Gain 999 gold. ${curse("2 Normality")}`,
      "i am healthy": () => `Heal to full HP. ${curse("Doubt")}`,
    },
  },
  MoaiHead: {
    about: "A giant stone head with an open mouth.",
    options: {
      "jump in": (e) => `Lose ${e.n[0]} max HP. Heal to full HP.`,
      "offer golden idol": () => "Give Golden Idol. Gain 333 gold.",
    },
  },
  MysteriousSphere: {
    about: "A mysterious sphere, guarded.",
    options: {
      "open the sphere": () => "Fight 2 Orb Walkers. Win: 45-55 gold and a rare relic.",
      fight: () => "Fight 2 Orb Walkers. Win: 45-55 gold and a rare relic.",
    },
  },
  SecretPortal: { about: "A portal that leads straight to the boss.", options: { "enter the portal": () => "Go straight to the boss." } },
  SensoryStone: {
    about: "A stone that brings back memories.",
    options: {
      "recall: 1": () => "Choose a colorless card.",
      "recall: 2": () => "Choose 2 colorless cards. Lose 5 HP.",
      "recall: 3": () => "Choose 3 colorless cards. Lose 10 HP.",
    },
  },
  TombRedMask: {
    about: "The tomb of the Red Mask's maker.",
    options: {
      "don the mask": () => "Gain 222 gold. Needs Red Mask.",
      "pay all gold": (e, o) => `Lose all your gold (${o.gold}). Obtain Red Mask.`,
    },
  },
  WindingHalls: {
    about: "Halls that twist back on themselves.",
    options: {
      "embrace madness": (e) => `Lose ${e.n[0]} HP. Obtain 2 Madness.`,
      retrace: (e) => `Heal ${e.n[1]} HP. ${curse("Writhe")}`,
      "press on": (e) => `Lose ${e.n[2]} max HP.`,
    },
  },
};

// The game's event keys (the view's `id`) where they differ from the names above.
const BY_KEY = { "Big Fish": "BigFish", "The Cleric": "Cleric", "Dead Adventurer": "DeadAdventurer", "Golden Idol": "GoldenIdol", "Golden Wing": "GoldenWing", "World of Goop": "WorldOfGoop", "Liars Game": "LiarsGame", "Living Wall": "LivingWall", "Scrap Ooze": "ScrapOoze", "Shining Light": "ShiningLight", "Match and Keep!": "MatchAndKeep", "Golden Shrine": "GoldenShrine", "Transmorgrifier": "Transmogrifier", "Upgrade Shrine": "UpgradeShrine", "Wheel of Change": "WheelOfChange", "Accursed Blacksmith": "AccursedBlacksmith", "Bonfire Elementals": "Bonfire", "Fountain of Cleansing": "FountainOfCleansing", "The Woman in Blue": "WomanInBlue", "Back to Basics": "BackToBasics", "Cursed Tome": "CursedTome", "Drug Dealer": "DrugDealer", "Forgotten Altar": "ForgottenAltar", "Masked Bandits": "MaskedBandits", "The Library": "TheLibrary", "The Mausoleum": "TheMausoleum", "Knowing Skull": "KnowingSkull", "N'loth": "Nloth", "The Joust": "TheJoust", "The Moai Head": "MoaiHead", "Mysterious Sphere": "MysteriousSphere", "Tomb of Lord Red Mask": "TombRedMask", "Winding Halls": "WindingHalls" };
const textFor = (e) => EVENT_TEXT[BY_KEY[e.id] || e.id];

/// What option `label` of event view `e` does, or "" (the option's name says it all, or the text
/// needs numbers this engine's view doesn't have). `o`: the run's view; `a`: the ascension.
export function eventOutcome(e, label, o, a) {
  const t = textFor(e);
  const l = String(label).toLowerCase();
  if (!t) return "";
  const key = Object.keys(t.options).filter((k) => l.startsWith(k)).sort((x, y) => y.length - x.length)[0];
  // An engine from before the event's numbers were in its view (an older network's) has no `n`.
  if (key === undefined || (!e.n && /\be\.n\b/.test(String(t.options[key])))) return "";
  return t.options[key]({ stage: 0, ...e }, o, a) || "";
}

/// The line under the event's title, or "".
export function eventAbout(e) {
  const t = textFor(e);
  if (!t) return "";
  return typeof t.about === "function" ? t.about({ stage: 0, n: [0, 0, 0, 0], ...e }) : t.about;
}
