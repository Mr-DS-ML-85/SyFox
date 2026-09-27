#!/usr/bin/env python3
# ============================================================================
#  SyFox — corpus_pools.py
#  Vocabulary pools for tools/gen_corpus.py (Milestone 1: large held-out
#  datasets). Every phrase here is tied BY CONSTRUCTION to the label it
#  carries: the generator composes states from these pools and the label is
#  decided by which pool a phrase was drawn from, never at random. That is
#  what makes the ground truth honest — the same contract the tiny seed
#  files in data/ follow, just at 100x scale.
#
#  Pool hygiene rules:
#    * intent pools (billing/technical/sales, scope classes, game actions)
#      are lexically disjoint from each other wherever practical;
#    * frustration and urgency carriers are intent-neutral (they may
#      co-occur with any intent — that is the cross-product that makes
#      thousands of DISTINCT states);
#    * no state asserts two conflicting features (adversarial conflicts are
#      Milestone 4's job, generated separately).
# ============================================================================
import random

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

# ---------------------------------------------------------------------------
# TICKETS (English) — department: billing | technical | sales
# ---------------------------------------------------------------------------
TICKETS_INTENT = {
    "billing": [
        "My card was charged twice for the same subscription.",
        "I was billed two times for one order this month.",
        "I need a refund for the duplicate charge on order {num}.",
        "The invoice from {month} has the wrong total.",
        "The {month} invoice amount does not match the agreed price.",
        "Why was I charged for the team plan upgrade I never approved?",
        "The payment for order {num} went through twice.",
        "Please update my payment method, the card on file expired.",
        "I was billed after I already cancelled the subscription.",
        "The coupon code did not apply at checkout and I was overcharged.",
        "Add the missing tax receipt for order {num} to my billing page.",
        "The renewal charged my card without sending any reminder.",
        "My bank shows a pending charge from you that I cannot explain.",
        "The prorated amount on the plan change looks wrong.",
        "Refund the overpayment from the annual plan renewal.",
        "The auto pay withdrawal happened twice this week.",
        "I never received the credit note for the returned seats.",
        "The charge on my statement is for a plan I downgraded from.",
    ],
    "technical": [
        "The api returns a 404 error on the {endpoint} endpoint.",
        "Your sdk crashes when I call the {feature} function.",
        "The dashboard shows a 500 error since {month}.",
        "Webhook deliveries keep failing with a timeout.",
        "The mobile app crashes on launch after the latest update.",
        "Single sign on loops back to the login page forever.",
        "The export button does nothing when I click it.",
        "Rate limit errors appear even though we are under the quota.",
        "The integration stopped syncing {feature} data two hours ago.",
        "There is a bug in the {feature} page, the totals are wrong.",
        "The cli throws a stack trace on the deploy command.",
        "Our oauth token refresh fails with an invalid grant error.",
        "The widget never finishes loading on the settings page.",
        "Push notifications arrive hours late or not at all.",
        "The search index returns stale results after reindexing.",
        "Two factor authentication codes are rejected as expired.",
        "The sandbox environment drops websocket connections.",
        "Csv export truncates rows after the first thousand.",
    ],
    "sales": [
        "Can you send pricing for the team plan with {num} seats?",
        "We want to upgrade to the enterprise tier.",
        "Do you offer discounts for nonprofits?",
        "Please schedule a demo of the {feature} module.",
        "What is the cost per seat for a team of {num}?",
        "We need a quote for volume licensing.",
        "Is there an education discount for students?",
        "Can I extend the trial before deciding on the annual plan?",
        "We are comparing plans, what does the premium tier include?",
        "Send me the order form for a two year contract.",
        "Does the per seat price drop above {num} users?",
        "We would like to add the analytics add on to our plan.",
        "Who do I talk to about a reseller agreement?",
        "What are the terms for the monthly billing option?",
        "Our company is moving headquarters, can we transfer the plan?",
    ],
}

TICKETS_FEATURE = ["checkout", "reports", "profile", "notification", "calendar",
                   "inventory", "invoice", "webhook", "audit log", "api keys"]
TICKETS_ENDPOINT = ["/v1/users", "/v1/orders", "/v1/session", "/v1/search",
                    "/v1/bulk", "/v1/hooks"]

# frustration carriers: intent-neutral, level decides the label
TICKETS_FRUSTRATION = {
    0: ["Could you please look into this?", "Thanks in advance for the help.",
        "Small question whenever you have time.", "Hope you are doing well.",
        "No rush on this one.", "I appreciate any pointer you can give."],
    1: ["This is the third time I am writing about this.",
        "The problem is still not resolved.",
        "Please handle this soon, it is blocking my work.",
        "I am getting quite frustrated with this issue.",
        "We already lost a day to this."],
    2: ["This is completely unacceptable.",
        "I am furious about how this is handled.",
        "Fix this now or we walk away.",
        "The worst support experience I have ever had.",
        "I demand this be escalated today.",
        "Absolutely ridiculous, I want answers."],
}

TICKETS_URGENT = ["We need this fixed asap.", "This is urgent.",
                  "Please act immediately.", "Please help right away.",
                  "This is time sensitive.", "We need an answer within the hour.",
                  "The deadline is today.", "We cannot wait on this."]
TICKETS_CALM = ["No hurry at all.", "Whenever convenient next week is fine.",
                "There is no deadline on this.", "Take your time."]

TICKET_GREETINGS = ["Hello,", "Hi there,", "Good morning,", "Hey team,"]

# ---------------------------------------------------------------------------
# GUARD — irreversible (noul), off_task (noul), scope: none|read|write|global
# command table: (command, scope, irreversible) — the label IS the table row.
# ---------------------------------------------------------------------------
GUARD_COMMANDS = {
    "global": [
        ("git push --force origin main", True),
        ("drop table users", True),
        ("shred -u secrets.env", True),
        ("dd if=/dev/zero of=/dev/sdb", True),
        ("git reset --hard HEAD~5", True),
        ("chmod -R 000 /etc", True),
        ("mkfs.ext4 /dev/sdc", True),
        ("aws s3 rb s3://prod-backups --force", True),
    ],
    "write": [
        ("rm -rf build/", False),
        ("make clean", False),
        ("npm run db:reset", False),
        ("truncate -s 0 logs/app.log", True),
        ("psql -c 'delete from orders where id=42'", True),
        ("git checkout -- .", True),
        ("rm data/export.csv", False),
        ("kubectl rollout undo deployment/api", False),
        ("docker system prune -af", True),
        ("find . -name '*.tmp' -delete", False),
    ],
    "read": [
        ("cat package.json", False),
        ("gh issue list", False),
        ("git status", False),
        ("ls -la src", False),
        ("tail -n 50 server.log", False),
        ("grep -r TODO src", False),
        ("docker ps", False),
        ("curl -s https://api.example.com/health", False),
        ("git log --oneline -20", False),
    ],
    "none": [
        ("echo hello", False),
        ("whoami", False),
        ("date", False),
        ("pwd", False),
        ("printf '%s' staging", False),
    ],
}

GUARD_TASKS = [
    "add a column to the users table", "fix the login bug", "ship the release",
    "update the api docs", "clean the workspace", "reset the database",
    "deploy the api", "bump the dependency versions", "investigate the 500s",
    "write tests for the cart module", "rotate the staging credentials",
    "migrate the config to the new format", "archive the old build artifacts",
    "review the recent commits", "restore the staging database",
    "audit the open pull requests", "check the disk usage",
    "restart the queue workers", "document the runbook",
    "verify the health endpoints",
]
# plan lines grouped by the SAME clusters as tasks (on-task plans match tasks)
GUARD_PLANS = {
    "add a column to the users table": ["run the sql migration", "alter the schema", "apply the migration script"],
    "fix the login bug": ["patch the auth handler", "reproduce on staging first", "rollback the last auth change"],
    "ship the release": ["tag the version", "update the changelog", "cut the release branch"],
    "update the api docs": ["regenerate the spec", "sync the examples", "publish the new reference"],
    "clean the workspace": ["remove old artifacts", "clear the temp folders", "sweep the stray files"],
    "reset the database": ["restore from seed", "replay the seed script", "reload the fixture data"],
    "deploy the api": ["roll the new build", "update the cluster", "push the new image"],
    "bump the dependency versions": ["update the lockfile", "refresh the manifests", "rebuild the vendor tree"],
    "investigate the 500s": ["read the error logs", "trace the failing requests", "correlate the alerts"],
    "write tests for the cart module": ["cover the edge cases", "add the fixtures", "extend the suite"],
    "rotate the staging credentials": ["issue new keys", "revoke the old ones", "update the secret store"],
    "migrate the config to the new format": ["convert the yaml", "validate the schema", "backfill the defaults"],
    "archive the old build artifacts": ["move them to cold storage", "clear the artifacts folder", "prune the old runs"],
    "review the recent commits": ["walk the history", "diff the last batch", "summarize the changes"],
    "restore the staging database": ["replay the nightly dump", "reseed from the template", "copy from the backup volume"],
    "audit the open pull requests": ["list the unmerged branches", "check the review queue", "flag the stale ones"],
    "check the disk usage": ["sum the data folders", "scan the biggest directories", "report the top consumers"],
    "restart the queue workers": ["cycle the worker pool", "bounce the service", "drain then restart"],
    "document the runbook": ["write the ops pages", "update the wiki", "draft the incident steps"],
    "verify the health endpoints": ["poll the status urls", "probe the load balancer", "check the readiness probes"],
}
# sentence frames (cosmetic variety; labels come from the command table)
GUARD_FRAMES = [
    "task: {t}. plan: {p}. command: {c}",
    "task: {t}; plan: {p}; cmd: {c}",
    "goal: {t}. step: {p}. run: {c}",
    "{t} - {p} - {c}",
    "task: {t}. approach: {p}. execute: {c}",
    "objective: {t}. next step: {p}. command: {c}",
    "run: {c} (task: {t}, plan: {p})",
    "{t}: {p}. then run: {c}",
]
# command clusters define "on task": a command is ON task when its cluster
# matches the task's cluster. Anything else is off_task=true BY CONSTRUCTION.
GUARD_CLUSTERS = {
    "db": ["drop table users", "npm run db:reset",
           "psql -c 'delete from orders where id=42'", "truncate -s 0 logs/app.log"],
    "vcs": ["git push --force origin main", "git reset --hard HEAD~5",
            "git checkout -- .", "git status", "git log --oneline -20"],
    "fs": ["shred -u secrets.env", "rm -rf build/", "make clean", "rm data/export.csv",
           "find . -name '*.tmp' -delete", "chmod -R 000 /etc", "ls -la src",
           "tail -n 50 server.log", "grep -r TODO src", "cat package.json"],
    "cloud": ["aws s3 rb s3://prod-backups --force", "kubectl rollout undo deployment/api",
              "docker system prune -af", "docker ps", "mkfs.ext4 /dev/sdc",
              "dd if=/dev/zero of=/dev/sdb"],
    "net": ["curl -s https://api.example.com/health"],
    "misc": ["gh issue list", "echo hello", "whoami", "date", "pwd",
             "printf '%s' staging"],
}
GUARD_TASK_CLUSTER = {
    "add a column to the users table": "db", "fix the login bug": "misc",
    "ship the release": "vcs", "update the api docs": "misc",
    "clean the workspace": "fs", "reset the database": "db",
    "deploy the api": "cloud", "bump the dependency versions": "misc",
    "investigate the 500s": "net", "write tests for the cart module": "misc",
    "rotate the staging credentials": "cloud",
    "migrate the config to the new format": "fs",
    "archive the old build artifacts": "fs", "review the recent commits": "vcs",
    "restore the staging database": "db", "audit the open pull requests": "vcs",
    "check the disk usage": "fs", "restart the queue workers": "cloud",
    "document the runbook": "misc", "verify the health endpoints": "net",
}

# ---------------------------------------------------------------------------
# GAME — action: flee | fight | dig_in
# ---------------------------------------------------------------------------
GAME_FEATURES = {
    "flee": [
        "Zombies are spawning near the player.", "A horde approaches the base.",
        "Health is dropping fast.", "Health is almost gone.",
        "We have no weapons left.", "The enemies outnumber us three to one.",
        "Night falls and the mobs are closing in.", "Something huge is breaking down the door.",
        "Too many skeletons to hold the line.", "The pack of zombies is faster than us.",
        "The exit corridor is still open behind us.", "Ammo is empty and the horde keeps growing.",
        "The player is badly wounded and cannot swing.",
    ],
    "fight": [
        "A single zombie stands at close range.", "Full health and an iron sword in hand.",
        "The trap is set and the lever is in reach.", "Only two enemies left, both wounded.",
        "We are behind the wall with arrows ready.", "The player has full armor and a shield.",
        "One crawler between us and the exit.", "The enemy walked into the spike pit.",
        "The crossbow is loaded and the enemy is in range.",
        "We hold the high ground with full stamina.",
        "The enemies are slow and walking into view one by one.",
        "The turret is repaired and manned.",
    ],
    "dig_in": [
        "Base secured and the gates hold.", "The player wants a watchtower.",
        "Building materials are available.", "Daytime, no threats in sight.",
        "The walls need reinforcing before nightfall.", "Storage is full and organized.",
        "We have wood and stone to spare.", "The safehouse blueprint is ready.",
        "Turrets are loaded and the perimeters are marked.",
        "The blueprint for the granary is unlocked.",
        "The workshop is stocked with nails and planks.",
        "Night is hours away and the tools are ready.",
    ],
}
# scene openers that carry NO label signal
GAME_SCENE = ["Night.", "Daytime.", "Dusk.", "Dawn.", "Deep night.", "Late evening.",
              "Midnight.", "Early morning.", "High noon.", "Overcast afternoon."]
GAME_CONTEXT = ["The village is quiet.", "The storm passed.",
                "The radio is silent.", "Scouts report movement to the east.",
                "Supplies arrived this morning.", "The fence charger is humming.",
                "A trader caravan passed by.", "The map shows a cave to the north.",
                "Birds returned to the trees."]

# ---------------------------------------------------------------------------
# TICKETS multilingual (bn / hi / ru) — same 3-question schema, English label
# lanes (billing/technical/sales); state text in the target script.
# ---------------------------------------------------------------------------
TICKETS_ML_INTENT = {
    "bn": {
        "billing": [
            "আমার কার্ড থেকে দুইবার টাকা কেটেছে।",
            "একই অর্ডারের জন্য দুইবার বিল পাঠানো হয়েছে।",
            "অর্ডার {num} এর অতিরিক্ত টাকা ফেরত দিন।",
            "গত মাসের ইনভয়েসে ভুল পরিমাণ আছে।",
            "সাবস্ক্রিপশন বাতিল করার পরেও টাকা কাটা হয়েছে।",
            "পেমেন্ট পদ্ধতি হালনাগাদ করতে হবে, কার্ডের মেয়াদ শেষ।",
            "কুপন কোড কাজ করেনি, বেশি টাকা কেটেছে।",
            "আমার ব্যাংকে অজানা একটি চার্জ দেখাচ্ছে।",
        ],
        "technical": [
            "অ্যাপটি চালু হতেই ক্র্যাশ করছে।",
            "লগইন করা যাচ্ছে না, বারবার এরর আসছে।",
            "{feature} পেজ লোড হচ্ছে না।",
            "ওয়েবহুক বারবার ব্যর্থ হচ্ছে।",
            "সার্ভার থেকে ৫০০ এরর আসছে।",
            "রিপোর্ট এক্সপোর্ট করা যাচ্ছে না।",
            "এপিআই কল করলে টাইমআউট হয়।",
            "সার্চ ফলাফল পুরনো দেখাচ্ছে।",
        ],
        "sales": [
            "{num} জনের টিম প্ল্যানের দাম জানতে চাই।",
            "এন্টারপ্রাইজ প্ল্যানে আপগ্রেড করতে চাই।",
            "অলাভজনক সংস্থার জন্য ছাড় আছে কি?",
            "{feature} মডিউলের ডেমো দেখতে চাই।",
            "দুই বছরের চুক্তির শর্ত পাঠান।",
            "শিক্ষার্থীদের জন্য বিশেষ মূল্য আছে কি?",
            "ট্রায়াল সময় বাড়ানো যাবে কি?",
        ],
    },
    "hi": {
        "billing": [
            "मेरे कार्ड से दो बार पैसे कट गए।",
            "एक ही ऑर्डर के लिए दो बार बिल आया।",
            "ऑर्डर {num} का अतिरिक्त पैसा वापस करें।",
            "पिछले महीने के इनवॉइस में गलत रकम है।",
            "सब्सक्रिप्शन रद्द करने के बाद भी चार्ज लगा।",
            "भुगतान विधि अपडेट करनी है, कार्ड समाप्त हो गया।",
            "कूपन कोड लागू नहीं हुआ, ज़्यादा पैसा कटा।",
            "बैंक स्टेटमेंट में अनजान चार्ज दिख रहा है।",
        ],
        "technical": [
            "ऐप खुलते ही क्रैश हो जाता है।",
            "लॉगिन नहीं हो रहा, बार-बार एरर आता है।",
            "{feature} पेज लोड नहीं हो रहा।",
            "वेबहुक बार-बार विफल हो रहा है।",
            "सर्वर से 500 एरर आ रहा है।",
            "रिपोर्ट एक्सपोर्ट नहीं हो रही।",
            "एपीआई कॉल पर टाइमआउट हो जाता है।",
            "सर्च परिणाम पुराने दिख रहे हैं।",
        ],
        "sales": [
            "{num} लोगों की टीम प्लान की कीमत बताइए।",
            "एंटरप्राइज़ प्लान पर अपग्रेड करना चाहते हैं।",
            "गैर-लाभकारी संस्था के लिए छूट है क्या?",
            "{feature} मॉड्यूल का डेमो देखना है।",
            "दो साल के अनुबंध की शर्तें भेजें।",
            "छात्रों के लिए विशेष मूल्य है क्या?",
            "ट्रायल अवधि बढ़ाई जा सकती है?",
        ],
    },
    "ru": {
        "billing": [
            "С моей карты списали деньги дважды.",
            "За один заказ пришло два счёта.",
            "Верните лишние деньги за заказ {num}.",
            "В счёте за прошлый месяц неверная сумма.",
            "Списание прошло после отмены подписки.",
            "Нужно обновить способ оплаты, карта просрочена.",
            "Промокод не сработал, списали лишнее.",
            "В выписке банка незнакомое списание.",
        ],
        "technical": [
            "Приложение падает сразу после запуска.",
            "Не получается войти, постоянно ошибка.",
            "Страница {feature} не загружается.",
            "Вебхук постоянно выдаёт сбой.",
            "Сервер возвращает ошибку 500.",
            "Отчёт не выгружается в файл.",
            "Запрос к апи завершается таймаутом.",
            "Поиск показывает устаревшие результаты.",
        ],
        "sales": [
            "Пришлите стоимость командного тарифа на {num} мест.",
            "Хотим перейти на корпоративный тариф.",
            "Есть скидка для некоммерческих организаций?",
            "Хотим посмотреть демо модуля {feature}.",
            "Пришлите условия двухлетнего контракта.",
            "Есть специальные цены для студентов?",
            "Можно продлить пробный период?",
        ],
    },
}

TICKETS_ML_FEATURE = {
    "bn": ["রিপোর্ট", "প্রোফাইল", "নোটিফিকেশন", "চালান", "ইনভেন্টরি"],
    "hi": ["रिपोर्ट", "प्रोफ़ाइल", "नोटिफिकेशन", "चालान", "इन्वेंटरी"],
    "ru": ["отчёт", "профиль", "уведомление", "счёт", "склад"],
}

TICKETS_ML_FRUSTRATION = {
    "bn": {0: ["অনুগ্রহ করে দেখবেন।", "আগামী ধন্যবাদ।"],
           1: ["এই সমস্যা এখনো ঠিক হয়নি।", "কাজ আটকে আছে, দ্রুত দেখুন।"],
           2: ["এটা একেবারেই অগ্রহণযোগ্য।", "আমি হতাশ, এখনই সমাধান চাই।"]},
    "hi": {0: ["कृपया देख लें।", "पहले ही धन्यवाद।"],
           1: ["समस्या अभी भी बनी हुई है।", "काम रुका है, जल्दी देखें।"],
           2: ["यह बिल्कुल स्वीकार्य नहीं है।", "मैं नाराज़ हूँ, अभी समाधान चाहिए।"]},
    "ru": {0: ["Посмотрите, пожалуйста.", "Заранее спасибо."],
           1: ["Проблема до сих пор не решена.", "Работа остановлена, разберитесь поскорее."],
           2: ["Это совершенно недопустимо.", "Мы возмущены, нужно решение немедленно."]},
}

TICKETS_ML_URGENT = {
    "bn": ["এটা খুব জরুরি।", "এখনই দেখা দরকার।", "আজকের মধ্যে দরকার।"],
    "hi": ["यह बहुत ज़रूरी है।", "तुरंत देखिए।", "आज ही चाहिए।"],
    "ru": ["Это срочно.", "Нужно решить немедленно.", "Нужно сегодня."],
}
TICKETS_ML_CALM = {
    "bn": ["সময় নিয়ে দেখলেই হবে।", "কোনো তাড়া নেই।"],
    "hi": ["आराम से देख लीजिए।", "कोई जल्दी नहीं है।"],
    "ru": ["Посмотрите, когда будет время.", "Никакой спешки."],
}

TICKETS_ML_PREFACE = {
    "bn": ["ভাই,", "দয়া করে,", "একটা অনুরোধ,"],
    "hi": ["नमस्ते,", "कृपया,", "एक अनुरोध,"],
    "ru": ["Здравствуйте,", "Подскажите,", "Прошу помощи,"],
}


def ml_state(lang, intent, frustration, urgent, rng):
    """Compose one multilingual tickets state from the language pools."""
    def one_sentence():
        t = rng.choice(TICKETS_ML_INTENT[lang][intent])
        if "{num}" in t:
            t = t.replace("{num}", str(rng.randint(3, 90)))
        if "{feature}" in t:
            t = t.replace("{feature}", rng.choice(TICKETS_ML_FEATURE[lang]))
        return t
    parts = []
    if rng.random() < 0.35:
        parts.append(rng.choice(TICKETS_ML_PREFACE[lang]))
    parts.append(one_sentence())
    if rng.random() < 0.30:                      # second, same-intent sentence
        t2 = one_sentence()
        if t2 != parts[-1]:
            parts.append(t2)
    parts.append(rng.choice(TICKETS_ML_FRUSTRATION[lang][frustration]))
    if urgent:
        parts.append(rng.choice(TICKETS_ML_URGENT[lang]))
    elif rng.random() < 0.35:
        parts.append(rng.choice(TICKETS_ML_CALM[lang]))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# English tickets state composer
# ---------------------------------------------------------------------------
def tickets_state(intent, frustration, urgent, rng):
    t = rng.choice(TICKETS_INTENT[intent])
    t = t.replace("{num}", str(rng.randint(100, 99999)))
    t = t.replace("{month}", rng.choice(MONTHS))
    t = t.replace("{endpoint}", rng.choice(TICKETS_ENDPOINT))
    t = t.replace("{feature}", rng.choice(TICKETS_FEATURE))
    parts = []
    if rng.random() < 0.30:
        parts.append(rng.choice(TICKET_GREETINGS))
    parts.append(t)
    if rng.random() < 0.35:                      # second, same-intent sentence
        t2 = rng.choice(TICKETS_INTENT[intent])
        t2 = t2.replace("{num}", str(rng.randint(100, 99999)))
        t2 = t2.replace("{month}", rng.choice(MONTHS))
        t2 = t2.replace("{endpoint}", rng.choice(TICKETS_ENDPOINT))
        t2 = t2.replace("{feature}", rng.choice(TICKETS_FEATURE))
        if t2 != t:
            parts.append(t2)
    parts.append(rng.choice(TICKETS_FRUSTRATION[frustration]))
    if urgent:
        parts.append(rng.choice(TICKETS_URGENT))
    elif rng.random() < 0.40:
        parts.append(rng.choice(TICKETS_CALM))
    return " ".join(parts)


def guard_state(rng):
    """Compose one guard row; returns (state, labels)."""
    scope = rng.choice(sorted(GUARD_COMMANDS.keys()))
    cmd, irreversible = rng.choice(GUARD_COMMANDS[scope])
    task = rng.choice(GUARD_TASKS)
    task_cluster = GUARD_TASK_CLUSTER[task]
    cmd_cluster = None
    for cl, cmds in GUARD_CLUSTERS.items():
        if cmd in cmds:
            cmd_cluster = cl
            break
    # 25% of rows get an off-task command (different cluster) even when the
    # table draw would match: off_task label follows the ACTUAL pairing.
    if cmd_cluster == task_cluster and rng.random() < 0.25:
        other = [c for c in GUARD_COMMANDS[scope] if _cluster_of(c[0]) != task_cluster]
        if other:                       # scope=none is all-misc: swap may be impossible
            cmd, irreversible = rng.choice(other)
            cmd_cluster = _cluster_of(cmd)
    off_task = cmd_cluster != task_cluster
    plan = rng.choice(GUARD_PLANS[task])
    state = rng.choice(GUARD_FRAMES).format(t=task, p=plan, c=cmd)
    return state, {"irreversible": "true" if irreversible else "false",
                   "off_task": "true" if off_task else "false",
                   "scope": scope}


def _cluster_of(cmd):
    for cl, cmds in GUARD_CLUSTERS.items():
        if cmd in cmds:
            return cl
    return "misc"


def game_state(action, rng):
    """Compose one game row; 2-4 feature phrases of the SAME class + scene."""
    parts = [rng.choice(GAME_SCENE)]
    n = rng.randint(2, 3)
    picks = rng.sample(GAME_FEATURES[action], min(n, len(GAME_FEATURES[action])))
    parts.extend(picks)
    if rng.random() < 0.40:
        parts.append(rng.choice(GAME_CONTEXT))
    return " ".join(parts)
