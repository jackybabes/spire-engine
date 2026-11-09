// ASCII figures for the fight: the four characters and every monster (by the game's monster id),
// drawn for this viewer (no game art). Bosses and elites are bigger. Each is a few lines of
// monospace text; `art(id)` gives the lines without the leading blank line.

const fig = (s) => s.replace(/^\n/, "").replace(/\s+$/, "");

export const PLAYER_ART = {
  ironclad: fig(String.raw`
    ___
   [o_o]
  /|###|\__
   |###|  |
   /   \  =
  /_   _\ `),
  silent: fig(String.raw`
     /\
    /..\
   /|><|\
  ~ |  | ~
   _/  \_ `),
  defect: fig(String.raw`
   [=-=]
  .-|o|-.
  | |=| |
  ' |_| '
   /   \ `),
  watcher: fig(String.raw`
    ___    |
   (===)   |
   /|_|\---|
  / |^| \  |
   /___\   | `),
};

export const MONSTER_ART = {
  // ---- Act 1 ----
  AcidSlime_L: fig(String.raw`
    .-~~~~-.
  .'  o  o  '.
 (     ~~     )
  '~-.____.-~' `),
  AcidSlime_M: fig(String.raw`
   .-~~-.
  ( o  o )
   '~--~' `),
  AcidSlime_S: fig(String.raw`
  .~~.
 (o o)
  '~' `),
  SpikeSlime_L: fig(String.raw`
   ^ ^ ^ ^ ^
  .-~~~~~~~-.
 (  o     o  )
  '~-.___.-~' `),
  SpikeSlime_M: fig(String.raw`
   ^ ^ ^
  ( o  o )
   '~--~' `),
  SpikeSlime_S: fig(String.raw`
  ^^^
 (o o)
  '~' `),
  JawWorm: fig(String.raw`
   ___
  /o  \____
 <VVV      >~
  \^^^/~~~~ `),
  Cultist: fig(String.raw`
    /\    *
   (oo)  /
  /|##|\/
   |##|
   /  \ `),
  FuzzyLouseNormal: fig(String.raw`
  \    /
  (o  o)
 /|####|\
   /  \ `),
  FuzzyLouseDefensive: fig(String.raw`
  \    /
  (o  o)
 [######]
   /  \ `),
  FungiBeast: fig(String.raw`
   .oOOo.
  (O ^^ O)
   \ __ /
   /|  |\ `),
  Looter: fig(String.raw`
   ___
  (-_-)  $
  /|=|\_/
   / \ `),
  Mugger: fig(String.raw`
   ___
  (>_<)
  /|=|\--
   / \ `),
  GremlinFat: fig(String.raw`
   /\_/\
  ( o o )
  (  ~  )
  (_____) `),
  GremlinThief: fig(String.raw`
  /\_/\
 ( -.- )>
  /| |\ `),
  GremlinTsundere: fig(String.raw`
  /\_/\
 ( >.< )[#]
  /| |\ `),
  GremlinWarrior: fig(String.raw`
  /\_/\
 ( \./ )
  /| |\= `),
  GremlinWizard: fig(String.raw`
   /\
  /__\
 ( o.o )*
  /| |\ `),
  GremlinNob: fig(String.raw`
   /\_____/\
  (  >   <  )
  /  _____  \
 | /|     |\ |
   /       \ `),
  Lagavulin: fig(String.raw`
   _________
  /  z  z   \
 |[=========]|
 |[=========]|
  \_________/ `),
  Sentry: fig(String.raw`
   _/\_
  |[ o ]|
  | ||| |
  |_____| `),
  SlaverBlue: fig(String.raw`
   ___
  (o_o)
  /|~|\~~@
   / \ `),
  SlaverRed: fig(String.raw`
   ___
  (o_o)
  /|~|\--<
   / \ `),
  Hexaghost: fig(String.raw`
   ( ~  ~  ~ )
  ~ .-------. ~
 ( |  O   O  | )
  ~ '-------' ~
   ( ~  ~  ~ ) `),
  SlimeBoss: fig(String.raw`
      .-~~~~~~~-.
    .'  O     O  '.
   (       ~~      )
   (      ____     )
    '~-.________.-~' `),
  TheGuardian: fig(String.raw`
   ____[O]____
  |  _|___|_  |
  |=|  ___  |=|
  | | |___| | |
  |_|_______|_| `),

  // ---- Act 2 ----
  Byrd: fig(String.raw`
    __
  <(o )___
   ( ._> /
    '---' `),
  Centurion: fig(String.raw`
   _/^\_
  |[o o]|
  /|###|\[]
   /   \ `),
  Healer: fig(String.raw`
    _
   (o)  +
  /|~|\/
   / \ `),
  Chosen: fig(String.raw`
    /\/\
   ( oo )
  /|\  /|\
    /  \ `),
  "Shelled Parasite": fig(String.raw`
   _______
  /#######\
 (  o   o  )~
  \_______/ `),
  SnakePlant: fig(String.raw`
   \  |  /
  (o)(o)(o)
   \ || /
  ___||___ `),
  Snecko: fig(String.raw`
   _______
  ( O   O )
  (  ___  )~~
   \_____/ `),
  SphericGuardian: fig(String.raw`
   .---.
  ( [o] )
   '---' `),
  BanditBear: fig(String.raw`
  (\_/)
  (o o)
  (=Y=)
  /   \ `),
  BanditChild: fig(String.raw`
   /\
  (oo)
  /||\-->
   /\ `),
  BanditLeader: fig(String.raw`
   ___
  (^_^)
  /|=|\~
   / \ `),
  BookOfStabbing: fig(String.raw`
   ________
  |\ \ \ \ |
  |  o  o  |-+-
  |________| `),
  GremlinLeader: fig(String.raw`
   _/\_/\_
  ( o v o )
  /|=WWW=|\
   /     \ `),
  SlaverBoss: fig(String.raw`
   _____
  ( o_o )
  /|$$$|\~~
   |___|
   /   \ `),
  TorchHead: fig(String.raw`
   (\~/)
   ( * )
    |=| `),
  BronzeAutomaton: fig(String.raw`
   ___[ ]___
  |  O   O  |
  |  =====  |
 [|  |___|  |]
   /       \ `),
  BronzeOrb: fig(String.raw`
  .---.
 ( (o) )
  '---' `),
  Champ: fig(String.raw`
    _/\_
   ( >< )
  /|=##=|\
 [] |##| []
    /  \ `),
  TheCollector: fig(String.raw`
    .~~~~~.
   ( O   O )
  ~/|~~~~~|\~
   (_______) `),

  // ---- Act 3 ----
  Darkling: fig(String.raw`
   /\  /\
  ( o  o )
   \/\/\/ `),
  Exploder: fig(String.raw`
   \ | /
  -( * )-
   / | \ `),
  Repulsor: fig(String.raw`
   ._.
  ( o )
  /'-'\ `),
  Spiker: fig(String.raw`
  ^^^^^
 <( o )>
  vvvvv `),
  Maw: fig(String.raw`
   _________
  / VVVVVVV \
 |           |
  \ ^^^^^^^ /
   '-------' `),
  "Orb Walker": fig(String.raw`
   .---.
  ( (O) )
   /| |\
  /     \ `),
  Serpent: fig(String.raw`
    __
  _(oo)
 ~  ||  ~
  \_/\_/ `),
  Transient: fig(String.raw`
  .~~~~.
 ( o  o )
  ~~~~~~
   ~ ~ `),
  WrithingMass: fig(String.raw`
  ~(o)~(o)~
 ~( ~~~~~ )~
  ~(o)~~(o)~ `),
  GiantHead: fig(String.raw`
   _______
  /       \
 |  O   O  |
 |    ^    |
  \ \___/ /
   '-----' `),
  Nemesis: fig(String.raw`
   _/^\_
  ( x x )
  /\~~~/\
 /  ~~~  \ `),
  Reptomancer: fig(String.raw`
    /\_
   ( o >~
  /|ss|\
   /  \ `),
  Dagger: fig(String.raw`
   /\
  (o>
  // `),
  AwakenedOne: fig(String.raw`
     \  |  /
    ( O   O )
   /|  ~~~  |\
  / |_______| \
     /     \ `),
  Donu: fig(String.raw`
   .-----.
  ( (   ) )
  ( ( o ) )
   '-----' `),
  Deca: fig(String.raw`
   [=====]
  [ [ o ] ]
   [=====] `),
  TimeEater: fig(String.raw`
    .-----.
   /  12   \
  | 9  o  3 |
  |    |    |
   \   6   /
    '-----' `),

  // ---- Act 4 ----
  SpireShield: fig(String.raw`
   _______
  |  ___  |
  | | o | |
  |  \_/  |
   \_____/ `),
  SpireSpear: fig(String.raw`
    /\
   /  \
  | o  |
   \  /
    || `),
  CorruptHeart: fig(String.raw`
   .-.   .-.
  (   \ /   )
   \   v   /
    \ ~~~ /
     \   /
      \ / `),
};

/// The figure for a player (`character`, any case) or a monster (its game id); "" when there is none.
export function artFor(id, player = false) {
  return (player ? PLAYER_ART[String(id).toLowerCase()] : MONSTER_ART[id]) || "";
}
