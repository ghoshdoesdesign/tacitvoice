"""Build sentences/pool.txt: a few thousand varied sentences for other people to read.

    python sentences/generate.py

When everyone reads the same few hundred sentences, a model can learn the
sentences instead of the mouth movements. This fills a large pool so that
each person can be given sentences almost nobody else reads. The sentences
are assembled from parts, so they are plainer than hand-written ones and
share a vocabulary of about 500 words; what varies is the combination. Half
are long, like real prompts.

The output is committed, and the same seed always gives the same file.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

HERE = Path(__file__).parent
COUNT = 3000
SEED = 7

TOPICS = """the history of chocolate | renewable energy | a healthy breakfast | remote work | the ocean floor |
ancient Rome | electric cars | urban gardening | the human heart | jazz music | deep sleep | honey bees |
the stock market | mountain weather | public libraries | the printing press | modern architecture |
childhood memories | coral reefs | the postal service | space travel | street food | volcanoes | chess openings |
the water cycle | handwriting | old maps | wind turbines | the silk road | bird migration | train travel |
small businesses | the night sky | bread baking | recycling plastic | forest fires | the common cold |
long distance running | paper money | lighthouses | the invention of the telephone | desert plants |
video games | the Olympic games | family recipes | river deltas | antique clocks | wild mushrooms |
city traffic | the northern lights | saving for retirement | learning the piano | glass making |
earthquakes | the wool trade | public speaking | whale songs | tea ceremonies | bicycle repair |
the first photograph | flu vaccines | apple orchards | the theory of evolution | folk tales | suspension bridges |
weather forecasts | indoor plants | the moon landing | fishing villages | school lunches | the color blue |
first aid | sourdough starters | medieval castles | solar panels | dog training | the rules of rugby |
river pollution | fashion in the twenties | honest feedback | camping in winter | the life of a butterfly |
online privacy | national parks | marble statues | good manners | coffee farming | house prices""".replace("\n", " ").split("|")

DOCS = """summary | short essay | blog post | checklist | speech | lesson plan | product description | cover letter |
press release | poem | short story | thank you note | travel guide | weekly schedule | shopping list | job advert |
newsletter | recipe | quiz | how to guide | status update | project brief | book review | welcome message |
birthday card | meeting agenda | training plan | budget | timeline | set of instructions""".replace("\n", " ").split("|")

ADJECTIVES = """short | clear | friendly | detailed | simple | playful | formal | honest | thoughtful | lively |
careful | practical | gentle | bold | calm | cheerful | polite | serious | vivid | modest | useful | brief |
warm | precise | balanced | curious | confident | patient | plain | fresh""".replace("\n", " ").split("|")

AUDIENCES = """a busy parent | my grandmother | a group of teenagers | new employees | a first year student |
my neighbours | a nervous beginner | the whole team | a small child | visiting tourists | my best friend |
a room of strangers | the school board | an impatient customer | my younger brother | a retired teacher |
local volunteers | a curious journalist | the village council | someone who has never tried it |
a group of engineers | my football club | the night shift | a class of ten year olds""".replace("\n", " ").split("|")

ACTIONS = """shorten | rewrite | translate | check | organise | simplify | expand | proofread | compare | review |
explain | summarise | improve | reorder | finish | update | measure | describe | test | tidy | merge | label |
sort | plan | sketch | fix | polish | double check | rethink | outline""".replace("\n", " ").split("|")

THINGS = """this paragraph | the opening line | my presentation | the final chapter | these instructions |
the monthly report | this spreadsheet | the invitation | my shopping list | the garden plan | this recipe |
the seating chart | my travel plans | the budget | this photograph | the welcome page | the second draft |
my notes from Tuesday | the kitchen layout | this contract | the team calendar | the packing list |
my reading list | the price table | this old letter | the workshop outline | the logo | the phone script |
my weekly routine | the floor plan""".replace("\n", " ").split("|")

QUALITIES = """shorter | clearer | warmer | more formal | less technical | easier to follow | more surprising |
more honest | funnier | more precise | less wordy | more hopeful | calmer | more direct | friendlier |
more persuasive | less dramatic | simpler | more vivid | better organised""".replace("\n", " ").split("|")

CONSTRAINTS = """before lunch | in plain language | without using jargon | in under a minute | by Friday evening |
for free | with only five ingredients | using everyday words | in two sentences | one step at a time |
without spending much money | while keeping the same meaning | in a way children would enjoy |
before the shops close | with examples from daily life | as quietly as possible | from memory |
in a single afternoon | without any special tools | so that anyone can follow it""".replace("\n", " ").split("|")

QUESTIONS = """What is the best way to learn about | Why do so many people care about | How did we first discover |
Who should I ask about | Where can I read more about | What surprised you most about | How would you explain |
Is there a simple rule for understanding | What do experts still disagree on about | When did people start studying |
What is a common mistake people make with | Why is it so hard to agree about | How much does it cost to explore |
Which book would you recommend on | What would a child ask about""".replace("\n", " ").split("|")

VERBS = """Write | Draft | Prepare | Sketch out | Put together | Give me | Create | Suggest | Compose | Outline""".split("|")

FOLLOW_UPS = """Keep it {quality}, and make sure it works for {audience}. | Then tell me which part you would change first, and why. |
Use {topic} as the example all the way through. | After that, give me a second version that is {quality}. |
Do it {constraint}, and mention anything you are unsure about. | Start with the most important point and leave out the rest. |
Imagine you are explaining it to {audience} on a long train journey. | Please make the ending {quality} than the beginning. |
I would like it to feel {adjective} and {adjective2}, not clever. | Finish with one question that would make {audience} think.""".replace("\n", " ").split("|")

SHORT = """{verb} a {adjective} {doc} about {topic}. | {question} {topic}? | Can you {action} {thing} {constraint}? |
Please {action} {thing} and make it {quality}. | {verb} a {doc} for {audience}. | I need a {adjective} {doc} about {topic} {constraint}. |
Tell {audience} about {topic} {constraint}. | Make {thing} {quality} for {audience}. | {question} {topic}, in a few words? |
Could you {action} {thing} for {audience}? | {verb} three {adjective} ideas about {topic}. | Help me {action} {thing} {constraint}.""".replace("\n", " ").split("|")

LONG = """{verb} a {adjective} {doc} about {topic} for {audience}. {follow} | Can you {action} {thing} {constraint}? {follow} |
I am helping {audience} learn about {topic}, so {verb_lower} a {adjective} {doc} they can use {constraint}. |
{question} {topic}? {follow} | Please {action} {thing} so that it is {quality}, then {action2} {thing2} {constraint}. |
{verb} a {doc} about {topic} that is {adjective} and {adjective2}. {follow}""".replace("\n", " ").split("|")

VOCABULARY = {
    "topic": TOPICS, "doc": DOCS, "adjective": ADJECTIVES, "adjective2": ADJECTIVES, "audience": AUDIENCES,
    "action": ACTIONS, "action2": ACTIONS, "thing": THINGS, "thing2": THINGS, "quality": QUALITIES,
    "constraint": CONSTRAINTS, "question": QUESTIONS, "verb": VERBS,
}
ALLOWED = re.compile(r"^[A-Za-z' ,.?]+$")


def fill(template: str, rng: random.Random) -> str:
    """Fill a template's slots; a repeated slot gets the same word, numbered ones differ."""
    chosen: dict[str, str] = {}

    def pick(match: re.Match) -> str:
        slot = match.group(1)
        if slot == "follow":
            return fill(rng.choice(FOLLOW_UPS).strip(), rng)
        if slot == "verb_lower":
            return rng.choice(VERBS).strip().lower()
        if slot not in chosen:
            options = [w.strip() for w in VOCABULARY[slot]]
            taken = {chosen.get(slot.rstrip("2")), chosen.get(slot + "2")}
            chosen[slot] = rng.choice([w for w in options if w not in taken])
        return chosen[slot]

    return re.sub(r"\{(\w+)\}", pick, template)


def tidy(sentence: str) -> str:
    sentence = " ".join(sentence.split())
    # "a honest" -> "an honest", and so on, wherever a part begins with a vowel sound.
    sentence = re.sub(r"\b([Aa]) (?=[aeiou]|honest)", lambda m: m.group(1) + "n ", sentence)
    return sentence[0].upper() + sentence[1:]


def generate(count: int = COUNT, seed: int = SEED) -> list[str]:
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[str] = []
    while len(out) < count:
        templates = LONG if len(out) % 2 else SHORT
        sentence = tidy(fill(rng.choice(templates).strip(), rng))
        if sentence in seen or not ALLOWED.match(sentence):
            continue
        seen.add(sentence)
        out.append(sentence)
    return out


if __name__ == "__main__":
    lines = generate()
    header = "# Made by sentences/generate.py. Do not edit by hand: change the generator and run it again.\n"
    (HERE / "pool.txt").write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    lengths = sorted(len(line.split()) for line in lines)
    words = {w.lower().strip(",.?") for line in lines for w in line.split()}
    print(f"wrote {len(lines)} sentences, {lengths[0]}-{lengths[-1]} words (middle {lengths[len(lengths) // 2]}), "
          f"{len(words)} different words")
