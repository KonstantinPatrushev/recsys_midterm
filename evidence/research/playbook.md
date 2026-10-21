# sweet.tv (MAP@5): плейбук на оставшиеся сабмиты

Состояние на 21.10.2026, 19:32 MSK. В основе — только утверждения со статусом verified или corrected из проверок треков kaggle-general и recsys-competitions, плюс локальные файлы проекта.

**Чего не хватает во входе**
- Пришли 2 трека из 4. Треков 3–4 нет, поэтому проверенных статей по методам курса (ALS, ассоциативные правила, PPR, CBF) здесь нет (§9.6).
- Проверка recsys-competitions обрезана на пункте OTTO 3rd. Источники без полученного статуса опорой не служат (§9.5).
- Проверщику kaggle-general вход пришёл обрезанным на приёме «Final submission». Поэтому последний приём («Study analogous competitions…») не проверен и не используется.

**Обозначения:** V — verified; C — corrected (берётся исправленная формулировка); L — локальный файл проекта, прочитан при сборке плейбука (полные пути в §9.3); [n] — ссылки из §9.1–9.2.

**Приоритеты (по убыванию)**
1. Записать CV sub02 (5 сидов). Пересчитать OOF-CV при трёх тай-брейках. Оценить парную SE разницы sub01→sub02 и проверить, укладывается ли +0.0028 на тесте в ожидание (§2.5–2.6).
2. Выбрать тестовую историю A или B офлайн: adversarial validation плюс прокси на F2 (§2.4).
3. Проверить гибрид, а не только бейзлайны, на F0 и F2 (§2).
4. Разобрать ошибки на F1: что теряется на полноте, что на порядке, отдельно новые шоу и «нечистые» пользователи. Из этого — целевые кандидаты и признаки готовности (§3–4).
5. Собрать RRF-гибрид методов курса для отчёта (§4).
6. Финал — ансамбль с усреднением по сидам. Правило выбора зависит от ответа преподавателя (§5–6).

## 1. What we know about the competition

**Правила** (страницы Data и Evaluation, дамп Kaggle API [L1]; C)
- **Метка.**
  - Из сетки удалены `tv_show_id = 0`.
  - Для каждого пользователя и каждого показа screen time = просмотренное время / длительность показа. Учитываются показы с screen time ≥ 0.8.
  - Частота = число таких показов на пару (user, show).
  - Берутся строго 5 самых частых, у каждого пользователя решения ровно 5. Поэтому знаменатель AP@5 всегда равен 5 (apk делит на min(|actual|, k) [32], V).
- **Деление теста.** Public/private — 50/50 по пользователям, не по времени: 630 + 630 пользователей, оба на неделях 31–42.
- **Формат сабмита.** Строго 5 id через пробел, порядок важен.
- **Исходное соревнование.**
  - 10 сабмитов в день, дедлайн 21.11.2020 23:59 [L1].
  - Private: 1-е место 0.42912, 2-е 0.42905 (разница 0.00007). Public: 0.44297 и 0.44252 [L2] (V).
- **Файлы и колонки отличаются от описания:** `dataset11-30.csv` с колонками `stop_time` и `duraton`; `export_arh_11-20-final.csv`, `export_arh_21-30-final.csv`, `export_arh_31-42-final.csv` [L5].

**Данные** (L)
- В логе 4 914 пользователей и 202 канала, все 1 260 пользователей сабмита в нём есть [L4]; сессий 5.82 млн [L5].
- Пользователи сабмита смотрят больше остальных [L4] (V):
  - медиана времени 1 933 267 с против 1 224 337 с;
  - медиана активных недель 20 против 18;
  - в неделе 30 активны 96.5% против 91.7%.
- **Провал логов.**
  - Полных просмотров в неделях 25/26/27 — 13 770 / 9 983 / 10 978, в остальных неделях 11–30 — от 26 482 до 49 453 [L4] (C).
  - В цели F1 у 1 630 из 4 254 пользователей сессии есть менее чем в 11 из 12 недель [L15].
  - README проекта относит разрыв к неделям 20–27 [L18].
- **Ничьи на 5-м месте** в неделях 19–30 у 63% пользователей. Медианный счёт 1-го места — 9, 5-го — 2 [L4] (C).
- **Тай-брейк меняет CV одних и тех же предсказаний** (all / sub) [L6] (C):
  - id_asc — 0.41537 / 0.44419;
  - random — 0.390–0.391 / 0.421–0.427;
  - id_desc — 0.37329 / 0.41082.
- **Сессии внутри показа.** Суммирование даёт 615 979 полных просмотров, максимум — 562 939 [L5] (C). 379 228 сессий перекрывают предыдущую сессию того же пользователя [L4].
- **Часовые пояса** лога и сетки совпадают: совпадение концов сессий с концами передач максимально при сдвиге 0 [L7].
- **Каталог.** В сетке 31–42 — 6 646 шоу, 1 438 из них не выходили в неделях 11–30 [L4].
- **Метрика.** `src/metrics.py` повторяет семантику ml_metrics apk/mapk [32] (V). Встроенная метрика `map` в LightGBM делит на число позитивов среди строк-кандидатов и даёт 1.0 запросам без позитивов, то есть завышает результат [34] (V).

**Наши сабмиты** [L3] (L)

| # | Файл | CV F1 all / sub | Public | Private | (Pub+Priv)/2 ≈ весь тест |
|---|---|---|---|---|---|
| 1 | sub01_ranker_v1 | 0.41537 / 0.44419 | 0.44705 | 0.43278 | 0.43992 |
| 2 | sub02_hybrid_lr31_r250_s5 | ≈0.4235 / ≈0.4565 * | 0.44688 | 0.43848 | 0.44268 |

\* CV пятисидового sub02 в evidence не записан. В качестве прокси взят ENS_lr3_r250: 0.42352 / 0.45647 [L15].

**Что из этого следует**
- Прирост CV(sub) около +0.012 дал на всём тесте только +0.0028: на public −0.0002, на private +0.0057.
- Разрыв «CV(sub) − тест» вырос с 0.004 до примерно 0.014. Похоже, часть прироста на F1 — подгонка под F1: число листьев, раунды и признаки выбирались по тем же OOF.
- Один и тот же файл на двух половинах пользователей различается на 0.014 (sub01) и 0.008 (sub02). Шум на уровне одного файла — порядка 0.01. Для пары файлов разница на public и на private разошлась на 0.006.
- Оба файла выше 1-го места исходного соревнования и на public, и на private.

**Открытые вопросы**
1. **Тай-брейк организаторов.**
   - За id_asc: результат sub01 на всём тесте (0.4399) ближе к CV при id_asc (0.44419), чем при random или id_desc.
   - Свидетельство слабое. Если в неделях 31–42 нет провала логов (это неизвестно), ничьих там меньше, и CV занижен при любом тай-брейке.
   - Проверка: пересчитать OOF-CV конфигурации sub02 при трёх тай-брейках и сравнить с 0.44268.
2. **Сумма или максимум внутри показа**, и считать ли дважды перекрывающиеся сессии. Формулировка «time user watched the show» читается как сумма. Оставляем сумму, устойчивость моделей к max проверяем на CV.
3. **Тестовая история.** Вариант A — недели 15–19 + 28–30 (как в sub01 и sub02). Вариант B — недели 23–30 подряд, внутри них 25–27.
   - Полных просмотров: в истории F1 (недели 11–18) — 239 752, в A — 296 792 (+24%), в B — 230 708 (−4%) [L4].
   - По объёму B ближе к обучению, но провал затрагивает только часть пользователей. Решать офлайн (§2.4).
4. **Структура потерь на F1.** Какая доля меток не попала в кандидатов, а какая попала, но вне топ-5. Отдельно доли шоу, новых для пользователя и для каталога.
5. **Доля новых шоу в целевом окне** — в F1 (19–30 против 11–18) и в тесте (31–42 против 11–30). Если в тесте она больше, F1 недооценивает генераторы для новых шоу.
6. **Какой сабмит идёт в оценку:** лучший public, последний или выбранный. Спросить преподавателя (§6).

## 2. Validation design

**Фолды** (`src/data.py`; метка по официальному правилу, сумма, тай-брейк id_asc)

| Фолд | История | Цель | Назначение |
|---|---|---|---|
| F1 | 11–18 | 19–30: 12 недель, как тест; внутри провал | основной: выбор моделей, CV для отчёта, GroupKFold по user_id |
| F0 | 11–15 | 16–19: 4 недели, без провала | направление изменений |
| F2 | 11–27 (провал в истории) | 28–30: 3 чистые недели перед тестом | сдвиг во времени |
| Тест | 8 недель (A: 15–19 + 28–30) | 31–42 | длина истории как в F1 |

Цель 25–30 для проверки не брать: половина её приходится на провал (C).

1. **Метрика и популяция.**
   - Метрика — `mapk@5` из `src/metrics.py` по полному набору меток.
   - Основная популяция — пользователи сабмита, у которых в цели не меньше 5 «полных» шоу: на F1 это 1 199 из 1 260 [L4][L10]. Вторая — все 4 254 таких пользователя.
   - Решения принимать по CV на пользователях сабмита: у sub01 он 0.44419 при 0.4399 на тесте, тогда как CV по всем — 0.41537. Так же советуют источники: валидация должна быть похожа на тест ([18], [20], V; adversarial validation — [27], [28], V).
2. **Время и утечки.**
   - Только схема «история до cutoff → следующие недели»: [20] «Match your CV strategy to how the test data is structured»; [21] — временной сплит; [15] — «mimics the test data» (все V).
   - Всё, что обучается, строится внутри фолда: популярность, ALS, AR, PPR, кодировки, отбор признаков, гиперпараметры ([19]; [23] — nested CV; V).
3. **Окна признаков.**
   - Окна фиксированы относительно cutoff: последние 1/2/4/8 недель, с нормировкой на неделю.
   - Длина истории в обучении и на тесте одинаковая. Так делал Santander 2nd: тестовые признаки строились на том же числе лагов ради «similar feature interpretations» ([15], V).
   - Методы курса для тестовых признаков обучать на той же 8-недельной истории.
4. **Выбор тестовой истории A/B — офлайн.**
   - Adversarial validation: классификатор «строки F1-train против строк теста» на одних и тех же признаках, отдельно для A и для B. Меньший AUC — ближе к обучению; важности покажут, какие признаки сдвинуты ([27], [28], V).
   - Грубый прокси на F2: применить модель F1 к истории «последние 8 чистых недель» (12–19) и «8 недель подряд» (20–27), сравнить MAP@5 на неделях 28–30.
   - Отправлять B, только если обе проверки не против и ожидаемая разница больше шума.
5. **Шум и правило принятия изменения.**
   - Разброс сидов у lr_l31_r250: 0.4231 / 0.42109 / 0.42206 по всем, 0.45617 / 0.45337 / 0.45408 по пользователям сабмита [L15].
   - Парный бутстреп по пользователям признаёт значимой даже разницу сидов: seed1 − seed0 = −0.00182, CI95 [−0.00346, −0.00008] [L14]. Случайность обучения он не учитывает.
   - Принимать изменение при трёх условиях: на F1 CI95 не накрывает 0; прирост больше разброса сидов (около 0.002 по всем, 0.003 по сабмиту); на F0 и F2 прирост не меньше нуля. Сравнивать средние по 3 сидам.
   - На тесте ждать примерно четверть прироста F1 (оценка по одной паре файлов). Улучшения меньше примерно 0.005 на F1 отдельным сабмитом не проверять — собирать их в финальный файл.
6. **Сверка CV и LB при малом числе сабмитов.**
   - До отправки записать в evidence CV по всем, по сабмиту, на F0 и на F2. Для sub02 такого файла пока нет.
   - После отправки сравнить ΔLB с ΔCV в единицах парной SE: SE_diff = sd(ΔAP_u)/√630. sd считать по OOF F1 на пользователях сабмита, бутстрепом подвыборок по 630.
   - Решений по public не принимать ([19], [15], V; [5]: public лишь «only approximately equal» private, V). Private у late-сабмитов виден, но это те же недели и другая половина пользователей — только для калибровки.
7. **Определение метки.**
   - Правило официальное; свободны только сумма/максимум и тай-брейк.
   - Обучать на сумме и id_asc. CV в отчёте приводить при id_asc и при random. Изменение модели принимать, только если оно выигрывает в обоих вариантах.
   - Контроль провала: вариант метки F1 без недель 25–27 (19–24 + 28–30) не должен менять порядок моделей.
   - Не менять метку вопреки формулировке правила по одному результату LB. Собирать метку офлайн — как testset.py у OTTO ([31], V).

## 3. Candidate generation

**Сейчас** [L8][L12]
- Медиана — 79 кандидатов на пользователя. Полнота (доля шоу метки среди кандидатов) — 0.774 по всем, 0.818 по сабмиту.
- Top-20 от методов курса поднимают полноту до 0.795 / 0.834, но MAP@5 почти не меняют: по сабмиту 0.44803 → 0.44831.
- Когда меток пять, MAP@5 при идеальном порядке равен полноте. Потолок 0.818 при MAP около 0.456 значит, что главные потери — в порядке, а не в полноте.

**Источники кандидатов (по приоритету)**
1. **Повтор** — шоу, которые пользователь досматривал и которые выходят в целевом окне.
   - У H&M 1st это стратегия repurchase [9]. H&M 2nd добавлял к 130 кандидатам всё, что пользователь покупал [10] (V).
   - Локально REPEAT — 0.36044 / 0.39231 [L10].
2. **Привычка × сетка** — шоу в эфире целевого окна на каналах и в слотах пользователя. Счёт метки не может быть больше числа эфиров; `air_t_userch` — самый важный признак [L12].
3. **PPR** по графу user–show — лучший одиночный метод курса: при d = 0.7 — 0.36889 / 0.40386 [L10]. Графовых кандидатов брали H&M 2nd («random graph walk over item-user graph» [10]) и H&M 1st (ProNE, «about 0.002 up» [9]) (V).
4. **Item-CF и ассоциативные правила** по совместным просмотрам.
   - ItemCF у H&M 1st, cooccurrences у H&M 2nd (V).
   - У Volkovs соседские модели не хуже WRMF: Item-Item 0.1772 против 0.1641 по RPREC [1] (V).
   - Локально AR (support ≥ 20, агрегация max) — 0.31466 / 0.33587; max заметно лучше sum (0.31466 против 0.26554) [L10].
5. **ALS.**
   - С параметрами f64 / reg 0.05 / α 10 даёт 0.22312 — хуже популярности (0.27578). После подбора (f64, α 1, reg 10) — 0.35792 / 0.38401 [L10][L11].
   - В implicit `recommend()` по умолчанию стоит `filter_already_liked_items=True`, и уже просмотренное выбрасывается. Ставить False и передавать в `items=` шоу в эфире; вместе с `filter_items` этот параметр не работает [33] (V).
6. **Новые шоу** (1 438 id в 31–42 без истории [L4]): «наследник слота» и CBF по каналу, жанру и названию.
   - Layer6 2017: content similarity вместо interaction similarity [2] (V).
   - H&M 3rd: товары с тем же prod_name и jaccard по атрибутам [11] (V).
   - Квоту ограничить: у H&M 1st холодные товары в top-12 не попадают [9] (V).
   - CBF в одиночку слаб: 0.24346 / 0.25916 [L10].
7. **Популярность в эфире на каналах пользователя** — для добивки. У H&M 1st всем выдавали top-100 популярных [9] (V).

**Как мерить**
- `recall_at_n` для каждого генератора и для их объединения при N = 5/10/20/50/100. Полнота объединения — потолок MAP@5.
- Квоты генераторов подбирать под максимум попаданий при фиксированном общем числе. Так H&M 1st вручную подбирал N для каждой стратегии по HitNum@100 [9] (V). Полноту считать отдельно от качества ранжирования ([1]: 20K кандидатов с полнотой >90%, V).
- Разделить промахи F1 на «метки нет в кандидатах» и «метка в кандидатах, но вне топ-5».
- Срезы для анализа:
  - шоу, новое для пользователя или для каталога;
  - «чистые» пользователи: на 856 таких пользователях сабмита ENS_lr3_r250 даёт 0.49329 против 0.45647 у всех [L15];
  - децили активности.
  Анализ ошибок и трудных примеров — шаг рабочего цикла у bestfitting [18] (V).

## 4. Ranking model and features

**Обучающая таблица**
- Строки (user, кандидат) из F1 по 4 254 пользователям, у которых в метке не меньше 5 шоу. У sub02 — 417 915 строк [L17].
- Метка бинарная: 1, если шоу в топ-5 пользователя за недели 19–30. Градуированная метка хуже: 0.41129 против 0.41539 [L16].
- Позитивы, которых нет среди кандидатов, не добавлять — так делал H&M 3rd [11] (V).
- Обучать на всех пользователях. Обучение без «нечистых» хуже (0.41854), их недовзвешивание тоже хуже (0.42067), против 0.4231 при обучении на всех [L15].
- CV: GroupKFold по user_id → OOF → MAP@5 по полному набору меток ([14]: «We must train using GroupKFold»; [20]; V). Встроенную `map` использовать только для early stopping [34].
- Финал: дообучить на всех пользователях F1 с числом раундов, выбранным по OOF. Тест — 8 недель истории плюс сетка 31–42 ([20]: «Retrain on 100% of the data»; [15]: одинаковое число лагов; V).
- Дополнительные якоря (cutoff 14 или 16) — низкий приоритет: их метки накрывают провал и пересекаются с F1 по времени.
- Сырой `tv_show_id` в признаки не брать: явный id-признак дал +0.0009 по сабмиту и −0.0002 по всем [L6].

**Функция потерь** (F1, all / sub)
- lambdarank, 31 лист, 250 раундов: 0.4231 / 0.45617 — лучшая одиночная модель; на 150–300 раундах результат стоит на плато [L15].
- binary (31 лист, 250 раундов) — 0.41995 / 0.4522 [L15]. xendcg (31 лист) — 0.41955 / 0.45165, poisson по счёту — 0.41523 / 0.44543 [L14]. 127 листьев — 0.40978 [L13].
- У призёров единого ответа нет (все V):
  - H&M 1st: лучшая одиночная модель — классификатор LightGBM [9];
  - H&M 2nd: binary на этапе отбора, lambdarank на этапе ранжирования; lambdarankmap чуть лучше lambdarank [10];
  - H&M 3rd: XGBoost, CatBoost и LightGBM дают близкий CV [11];
  - Volkovs: ranking loss лучше BCE [1].
- Выбор по OOF MAP@5 уже сделан — lambdarank. Для перекрёстной проверки — XGBoost `rank:map` ([35], V).

**Признаки** ([+] — уже есть в sub02 [L17])
- [+] **Эфир в целевом окне** (будущую сетку использовать разрешено): эфиры шоу на каналах пользователя, всего эфиров, первые и последние 2–4 недели окна, дни недели, доля прайм-тайма, длительность, флаг нового шоу, отношение будущих эфиров к прошлым.
- [+] **Привычка к слоту:** ожидаемые полные просмотры по слоту и каналу (`exp_slot`, `exp_show_slot`, `exp_rate`) и их ранги.
- [+] **user × show:** полные просмотры за 1/2/4 недели и за всю историю, понедельные счётчики `f_w0..f_w7`, давность, частичные просмотры и near-miss, средний screen time, доля и ранг шоу у пользователя.
  - Опора: группы user / item / user × item / datetime и окна `_n5` у Instacart 2nd [16]; окна «неделя / месяц / сезон» у H&M 1st [9]; там же — «user and item interaction information are always the most important» [9] (V).
- **Готовность к повтору** — добавить, если анализ ошибок покажет, что выпадают устойчивые привычки:
  - недели с последнего полного просмотра минус обычный перерыв пользователя для этого шоу (аналог Days_last_order-max у Instacart [16], C);
  - «следующий просмотр ≈ последний + средний интервал» и streak (H&M 2nd [10], V).
- [+] **Популярность шоу:** зрители, полные просмотры, то же за последние 2 недели, доля. Добавить тренд — как weekly trend у H&M 3rd [11] (V).
- [+] **Пользователь** (объём, недели, каналы, доля главного канала) и **контент** (категория, жанр 1, год).
- [+] **Скоры и ранги методов курса** — PPR, AR, ALS, CBF. После добавления `m_ppr_rank` стал самым важным признаком [L12].
- **Флаги генераторов** — добавить: какой генератор предложил кандидата и сколько генераторов его предложили.
  - У H&M 3rd флаги и ранги стратегий вместе с расширением кандидатов подняли LB с 0.02855 до 0.03262; без них CV был «very very poor» [11] (V).
  - Volkovs подавал пять скоров первой стадии прямо во вторую [1] (V).

**Методы курса: требование «не меньше двух» выполнено тремя способами**
1. **Каждый метод отдельно** на одном сплите F1 и одной метрике [L10][L11] — таблица для отчёта:

| Метод | MAP@5 all | MAP@5 sub |
|---|---|---|
| POP (популярное в эфире) | 0.27578 | 0.29195 |
| CBF (raw) | 0.24346 | 0.25916 |
| AR (support ≥ 20, max) | 0.31466 | 0.33587 |
| ALS (f64, α 1, reg 10) | 0.35792 | 0.38401 |
| REPEAT | 0.36044 | 0.39231 |
| PPR (d = 0.7) | 0.36889 | 0.40386 |
| Гибрид LightGBM, ENS_lr3_r250 [L15] | 0.42352 | 0.45647 |

2. **Внутри гибрида:** скоры и ранги методов — признаки, их top-20 — кандидаты. Методы обучаются только на истории до cutoff, поэтому OOF им не нужен. Модели первого уровня, которые учатся на метках, — только с OOF по пользователям ([24], [26], V).
3. **Безмодельный гибрид** для сравнения в отчёте: RRF по рангам методов, score = Σ w_m / (k + rank_m), k = 60 («near-optimal, but … not critical» [3], V). Веса подобрать жадным перебором (hill climbing) по F1 ([20], [4], V) и сравнить с ранжировщиком на том же сплите.

## 5. Ensembling and post-processing for MAP@5

**Ансамбль**
- **Сейчас лучший** — 3 сида lambdarank (250 раундов): 0.42352 / 0.45647. Если добавить три binary-модели, становится хуже: 0.42267 / 0.45468 [L15].
- Прирост ансамбля над одной моделью (+0.00223, CI95 [0.00082, 0.00372] [L14]) — того же порядка, что разброс сидов. Ансамбль нужен, чтобы снизить дисперсию, а не поднять уровень.
- **Как смешивать.** Усреднять ранги внутри пользователя или нормированные скоры. Ранговое усреднение — MLWave [26]; у Santander 2nd «rank averaging typically works better than probability averaging» [15].
- **Как подбирать состав.** Добавлять участников по одному, пока растёт F1 и не падают F0 и F2 (hill climbing [20], ensemble selection [4]). Брать слабо коррелированные модели ([26]; «as uncorrelated as possible» [15]). Все V.
- **Сиды:** 3–5 (`seed`, `bagging_seed`, `feature_fraction_seed`); для воспроизводимых одиночных прогонов — `deterministic=True` [34]. Усреднение по сидам: в [20] — 100 сидов на задаче с MAP@3; у Instacart 2nd — шесть GBDT с разными сидами [16] (V).
- **Каких приростов ждать** (у призёров, по public LB, на других метриках): H&M 1st 0.0367 → 0.0371 [9]; H&M 2nd 0.0355 → 0.0362 → 0.0368 [10]; H&M, серебро — 0.0286 → 0.0292 [17] (V). Прирост небольшой, но стабильный.
- Блендинг на маленьком holdout не делать: «The final model may overfit to the holdout set» [26] (V).

**Постобработка**
- **Формат:** ровно 5 разных ненулевых id, 1 260 строк в порядке sample, все id в эфире 31–42. Это проверяет `tests/check_submission.py` [L19].
  - Дубликат в apk ничего не даёт [32] (V).
  - Пустых позиций не оставлять: на H&M «never a penalty for using the full 12 predictions» [13] (V).
- **Порядок** — по скору модели (вероятность попасть в топ-5), а не по ожидаемому числу просмотров: poisson по счёту хуже (0.41523) [L14].
- **Добивка.** У sub02 два пользователя без кандидатов добиты популярным, для восьми взята запасная история 11–18 [L17].
- **Перестановки под ожидаемый AP** — только если растут F1, F0 и F2. У Santander 2nd такая MAP-оптимизация дала «great effects in local validation (~0.2% boost) but limited value on the public leaderboard», а на private плюс был похожим ([15], C).

## 6. Submission budget plan

Сделано 2 сабмита из примерно 5, осталось 3 [L3].

В оценку идёт место по public среди одногруппников. Значит, для оценки public — это сама цель, а не шумная оценка качества. Public каждого отправленного файла известен, а файл воспроизводится побайтно [L19]. Для отчёта лучшая модель выбирается иначе — по CV и по (public + private)/2.

1. **Сейчас:** спросить преподавателя, какой сабмит считается.
2. **S3** — только структурное изменение, которое прошло правило §2.5 и даёт ожидаемый прирост на тесте больше шума парной разницы (около 0.005). Кандидаты:
   - тестовая история B или другая, если её поддержат офлайн-проверки §2.4;
   - находки анализа ошибок: кандидаты для новых шоу, признаки готовности.
   CV записать до отправки.
3. **S4** — финальный кандидат по CV: ансамбль с усреднением по сидам, обученный на всех пользователях F1.
4. **S5** — резерв, зависит от правила оценки:
   - если считается лучший public — ещё один честный вариант другой структуры; хуже не будет, лучший результат уже 0.44705;
   - если считается последний или выбранный — последним отправить (или указать) файл с наибольшим известным public, сейчас это sub01 с 0.44705.
   Bestfitting выбирает финал так же осторожно: «weighted average ensemble of my safe models» плюс один рискованный вариант, и никогда не берёт то, что не может объяснить [18] (V).
5. **Не делать** ([19], [15], [6], V):
   - отправлять два файла, у которых разница CV в пределах шума;
   - оставлять изменение только потому, что вырос public;
   - подбирать что-либо по private;
   - пробинг.
6. **Для отчёта:** таблица всех сабмитов с CV (по всем, по сабмиту, F0, F2), public и private; скриншот LB. Прямо написать, что финал для оценки выбран по public, потому что это и есть критерий оценки.

## 7. Pitfalls / leakage rules

1. **Правило курса.** Просмотры недель 31–42 не использовать ни в каком виде; сетку 31–42 — можно. На F1 «будущее» — недели 19–30: признаки по просмотрам этих недель запрещены, по их сетке — разрешены.
2. **Пробинг LB запрещён.** Сабмит «один id всем» вытаскивает метки 31–42, то есть будущее поведение пользователей.
   - Whitehill восстановил метки из оракула и занял 4-е место из 848, не обучив ни одного классификатора [7].
   - HSE относит «Leaderboard probing» к утечкам [30].
   - Blum & Hardt описывают переобучение под holdout лидерборда [6] (все V).
3. **Всё, что обучается, — внутри фолда**, включая отбор признаков и гиперпараметры. У Greg Park CV стал «increasingly optimistic», когда это делалось вне фолдов [19] (V). Похожий симптом у нас: на sub02 разрыв «CV − тест» вырос примерно до 0.014.
4. **Провал логов в неделях 20–27, глубже всего в 25–27.** В метке F1 из-за него меньше счётчиков и больше ничьих. На тесте не брать недели провала в окна без нормировки — вариант B (23–30) их содержит.
5. **Длина истории.** F1 видит 8 недель, тест — до 20. Без фиксированных окон меняется масштаб признаков [15] (V).
6. **Популяция.** Решения принимать по CV на пользователях сабмита: там 0.44419, а по всем — 0.41537.
7. **Тай-брейк id_asc** — гипотеза, а не факт.
8. **implicit:** по умолчанию `filter_already_liked_items=True` [33] (V); ALS без подбора параметров хуже популярности [L10][L11].
9. **Встроенная метрика `map` в LightGBM** завышает результат [34] (V).
10. **Затухание по времени здесь вредит.** DECAY_AIR с полупериодом 1 неделя даёт 0.29104 против 0.3493 у REPEAT_AIR (F1, по всем) [L9]. Метке за 12 недель нужна длинная память; рецепты H&M и OTTO с затуханием не переносить вслепую.
11. **Сырой `tv_show_id` как признак** запоминает период и пользы не даёт [L6].
12. **Public шумит.** Один и тот же файл на двух половинах даёт 0.447 и 0.433. 1-е и 2-е места исходного соревнования различаются на 0.00007, так что близкие места во многом случайны ([5]: «large rank changes can be a natural consequence of random noise», V). Порог Roelofs в 10 000 примеров выведен для соревнований на accuracy и к MAP@5 напрямую не переносится (C).
13. **Формат:** 1 260 строк в порядке sample, по 5 разных ненулевых id в эфире 31–42; в описании сабмита — ФИО, как у sub01 и sub02 [L3].

## 8. Tricks used, with sources

| Приём | Источник | URL | Статус |
|---|---|---|---|
| Валидация как тест: время, горизонт, популяция; GroupKFold | NVIDIA, The Kaggle Grandmasters Playbook (2025) | https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/ | V; TimeSeriesSplit и GroupKFold поданы как «Pro tip» |
| Валидация как тест; LB как проверка; анализ ошибок; осторожный финал | Profiling Top Kagglers: Bestfitting (Kaggle Blog, 2018), Wayback | https://web.archive.org/web/20180515173145id_/http://blog.kaggle.com/2018/05/07/profiling-top-kagglers-bestfitting-currently-1-in-the-world/ | V; в источнике «a period of time», не «a later period» |
| Временной сплит в рекомендательных задачах | NVIDIA, Co-Visitation Matrices and cuDF (Viel, 2024) | https://developer.nvidia.com/blog/build-efficient-recommender-systems-with-co-visitation-matrices-and-rapids-cudf/ | V |
| Собирать тест организаторов офлайн | otto-de/recsys-dataset, KAGGLE.md | https://github.com/otto-de/recsys-dataset/blob/main/KAGGLE.md | V |
| Правило метки sweet.tv; деление 50/50 по пользователям | Kaggle, sweet.tv – TV program recommender (Data/Evaluation, дамп API [L1]) | https://www.kaggle.com/competitions/sweettv-tv-program-recommender | C: меток строго 5; деление — официальное правило |
| Одинаковые окна и лаги в обучении и на тесте; сезонно выровненное окно; ранговое усреднение; MAP-оптимизация; вред отбора по public | Santander 2nd write-up (Van de Wiele, 2017), Wayback | https://web.archive.org/web/20170115012200id_/http://blog.kaggle.com:80/2017/01/12/santander-product-recommendation-competition-2nd-place-winners-solution-write-up-tom-van-de-wiele/ | V; C: MAP-оптимизация — «limited value» на public; 0.031 получено не на «top-10 features» |
| LB шумит на малых тестах | Roelofs et al., NeurIPS 2019 (PDF) | https://papers.nips.cc/paper_files/paper/2019/file/ee39e503b6bedf0c98c388b7e8589aca-Paper.pdf | V; порог 10 000 — для accuracy-соревнований |
| Бюджет сабмитов; всё внутри фолда; не доверять public | Greg Park, Kaggle postmortem (2012) | https://gregpark.io/blog/Kaggle-Psychopathy-Postmortem | V |
| Переобучение под holdout лидерборда | Blum & Hardt, The Ladder (arXiv) | https://arxiv.org/abs/1502.04585 | V |
| Без пробинга | Whitehill, log-loss oracle (arXiv) | https://arxiv.org/abs/1707.01825 | V |
| Пробинг как утечка; оптимизация под метрику; стекинг | HSE/Coursera, How to Win a Data Science Competition (Wayback, 2020) | https://web.archive.org/web/20200416102958id_/https://www.coursera.org/learn/competitive-data-science | V |
| Groupby-агрегации; nested CV для статистик по таргету | NVIDIA, Feature Engineering with cuDF pandas (Deotte, 2025) | https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas/ | V |
| Adversarial validation | FastML, Adversarial validation, part one (2016) | http://fastml.com/adversarial-validation-part-one/ | V |
| Adversarial validation (код) | The Kaggle Book repo, chapter_06 notebook | https://raw.githubusercontent.com/PacktPublishing/The-Kaggle-Book/main/chapter_06/adversarial-validation-example.ipynb | V; только репозиторий, сама книга — §9.4 |
| Adversarial validation для дрейфа | Pan et al., Uber (arXiv, 2020) | https://arxiv.org/abs/2004.03045 | V |
| Двухэтапная схема | NVIDIA, Multilingual Recommender Systems (Deotte, 2023) | https://developer.nvidia.com/blog/pro-tips-for-building-multilingual-recommender-systems/ | V |
| Двухэтапная схема; customer как query | H&M: Addressing common questions (Jankiewicz) | https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/307288 | V |
| GBT-ранжировщик: 50–200 кандидатов, GroupKFold, прореживание негативов | OTTO: How To Build a GBT Ranker Model (Deotte) | https://www.kaggle.com/competitions/otto-recommender-system/discussion/370210 | V |
| repurchase, ItemCF, популярность, графовые эмбеддинги → GBDT; HitNum@100; окна; хранилище признаков | H&M 1st place (senkin13 & 30CrMnSiA) | https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324070 | V |
| Схема пайплайна H&M 1st | H&M 1st, overview.png | https://i.postimg.cc/bwBm7GVv/overview.png | V |
| Повтор в кандидаты; следующая покупка; streaks; random graph walk; cooccurrences; binary → lambdarank | H&M 2nd place (wht1996 & Jankiewicz) | https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324197 | V; C: цитата про availability — из общего раздела команды |
| Флаги и ранги стратегий; BPR по неделям; окна 7/30/all; популярность с весом 1/days | H&M 3rd place (sirius) | https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324129 | V |
| MAP@K: заполнять все K позиций | H&M Evaluation | https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/overview/evaluation | V |
| Масштаб прироста от ансамбля | Wp-Zhang/H-M-Fashion-RecSys | https://github.com/Wp-Zhang/H-M-Fashion-RecSys | V; MAP@12 в README не указан |
| Группы признаков user / item / user×item; окна `_n5`; готовность к повтору; 6 сидов | Instacart 2nd (Onodera), Wayback | https://web.archive.org/web/20170922014019id_/http://blog.kaggle.com/2017/09/21/instacart-market-basket-analysis-winners-interview-2nd-place-kazuki-onodera/ | V; C: «how ready…» — это Days_last_order-max |
| Скоры первой стадии как признаки; соседские модели не хуже WRMF; ranking loss лучше BCE; полнота кандидатов | Volkovs et al., RecSys Challenge 2018 | https://www.cs.toronto.edu/~mvolkovs/recsys2018_challenge.pdf | V |
| CBF для холодных объектов | Volkovs, Yu, Poutanen, RecSys Challenge 2017 | https://www.cs.toronto.edu/~guangweiyu/pdfs/recsys2017.pdf | C: полное название — в §9.1 |
| ALS: не фильтровать просмотренное; `items=`; `random_state` | implicit docs, AlternatingLeastSquares | https://benfred.github.io/implicit/api/models/cpu/als.html | V |
| lambdarank и группы запросов; сиды; deterministic; встроенная map завышает | LightGBM Parameters; map_metric.hpp | https://lightgbm.readthedocs.io/en/latest/Parameters.html ; https://raw.githubusercontent.com/microsoft/LightGBM/master/src/metric/map_metric.hpp | V |
| rank:map | XGBoost, Learning to Rank | https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html | V |
| Семантика apk/mapk | benhamner/Metrics | https://raw.githubusercontent.com/benhamner/Metrics/master/Python/ml_metrics/average_precision.py | V |
| Ранговое усреднение; некоррелированные участники; минусы блендинга; OOF-стекинг | MLWave, Kaggle Ensembling Guide (2015), Wayback | https://web.archive.org/web/20160206102007id_/http://mlwave.com/kaggle-ensembling-guide/ | V |
| RRF, k = 60 | Cormack, Clarke, Büttcher, SIGIR 2009 (PDF) | http://cormack.uwaterloo.ca/cormack/cormacksigir09-rrf.pdf | V |
| Ensemble selection: жадное добавление моделей | Caruana et al., ICML 2004 (OpenAlex) | https://api.openalex.org/works/doi:10.1145/1015330.1015432 | V |
| Стекинг: OOF первого уровня как признаки второго | NVIDIA, Stacking with cuML (Deotte, 2025) | https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/ | V |
| Сохранять OOF и тестовые предсказания каждого эксперимента | NVIDIA, Generative AI-Assisted Coding (Deotte, 2026) | https://developer.nvidia.com/blog/winning-a-kaggle-competition-with-generative-ai-assisted-coding/ | V |
| GBDT доминирует на табличных данных | ML Contests, State of ML Competitions 2024 | https://mlcontests.com/state-of-machine-learning-competitions-2024/ | C: LightGBM — самая частая GBDT-библиотека, но не самая частая библиотека вообще |

## 9. Verified references for the report

### 9.1 Статьи и труды конференций (IEEE; строки от проверщика)
- [1] M. Volkovs, H. Rai, Z. Cheng, G. Wu, Y. Lu, and S. Sanner, "Two-stage model for automatic playlist continuation at scale," in Proc. ACM Recommender Systems Challenge 2018 (RecSys Challenge '18), Vancouver, BC, Canada, 2018, pp. 1–6, doi: 10.1145/3267471.3267480. — V (Crossref/OpenAlex; PDF https://www.cs.toronto.edu/~mvolkovs/recsys2018_challenge.pdf; номер статьи в ACM DL не проверен).
- [2] M. Volkovs, G. W. Yu, and T. Poutanen, "Content-based neighbor models for cold start in recommender systems," in Proc. Recommender Systems Challenge 2017 (RecSys Challenge '17), Como, Italy, 2017, pp. 1–6, doi: 10.1145/3124791.3124792. — C: в треке было обрезанное название (PDF https://www.cs.toronto.edu/~guangweiyu/pdfs/recsys2017.pdf).
- [3] G. V. Cormack, C. L. A. Clarke, and S. Büttcher, "Reciprocal rank fusion outperforms Condorcet and individual rank learning methods," in Proc. 32nd Int. ACM SIGIR Conf. Res. Develop. Inf. Retrieval (SIGIR '09), Boston, MA, USA, Jul. 2009, pp. 758–759, doi: 10.1145/1571941.1572114. — V.
- [4] R. Caruana, A. Niculescu-Mizil, G. Crew, and A. Ksikes, "Ensemble selection from libraries of models," in Proc. 21st Int. Conf. Mach. Learn. (ICML '04), Banff, AB, Canada, Jul. 2004, p. 18, doi: 10.1145/1015330.1015432. — V.
- [5] R. Roelofs, V. Shankar, B. Recht, S. Fridovich-Keil, M. Hardt, J. Miller, and L. Schmidt, "A meta-analysis of overfitting in machine learning," in Advances in Neural Information Processing Systems 32 (NeurIPS 2019), H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, Eds. Curran Associates, Inc., 2019, pp. 9179–9189. — C: страницы взяты из архивного BibTeX NeurIPS, OpenAlex даёт 9175–9185; в PDF другой порядок авторов (Roelofs*, Fridovich-Keil*, Miller, Shankar, Hardt, Recht, Schmidt); DOI у статьи нет. URL: https://papers.nips.cc/paper_files/paper/2019/file/ee39e503b6bedf0c98c388b7e8589aca-Paper.pdf
- [6] A. Blum and M. Hardt, "The Ladder: A reliable leaderboard for machine learning competitions," in Proc. 32nd Int. Conf. Mach. Learn. (ICML), Lille, France, Jul. 2015, vol. 37, Proc. Mach. Learn. Res., pp. 1006–1014. — V (https://proceedings.mlr.press/v37/blum15.html; arXiv:1502.04585).
- [7] J. Whitehill, "Climbing the Kaggle leaderboard by exploiting the log-loss oracle," in Workshops 32nd AAAI Conf. Artif. Intell., AAAI Tech. Rep. WS-18 (Engineering Dependable and Secure Machine Learning Systems), 2018; preprint arXiv:1707.01825, 2017. — V; страницы 392–399 известны только по OpenAlex, в PDF не проверены (https://cdn.aaai.org/ocs/ws/ws0421/16161-75972-1-PB.pdf).
- [8] J. Pan, V. Pham, M. Dorairaj, H. Chen, and J.-Y. Lee, "Adversarial validation approach to concept drift problem in user targeting automation systems at Uber," arXiv:2004.03045, 2020, doi: 10.48550/arXiv.2004.03045. — V (опубликован только препринт).

### 9.2 Веб-источники (в стиле IEEE; все открыты проверщиком; дату обращения добавить при цитировании)
- [9] senkin13 (команда с 30CrMnSiA), "1st place solution," H&M Personalized Fashion Recommendations, Kaggle discussion. [Online]. Available: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324070 (схема: https://i.postimg.cc/bwBm7GVv/overview.png) — V
- [10] wht1996 and P. Jankiewicz, "2nd place solution," H&M Personalized Fashion Recommendations, Kaggle writeup. [Online]. Available: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324197 (редирект на /writeups/hello-world-2nd-place-solution) — V
- [11] sirius, "3rd place solution," H&M Personalized Fashion Recommendations, Kaggle writeup, May 16, 2022. [Online]. Available: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324129 (редирект на /writeups/sirius-3rd-place-solution) — V
- [12] P. Jankiewicz, "Addressing common questions and what the competition is really about," H&M Personalized Fashion Recommendations, Kaggle discussion. [Online]. Available: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/307288 — V
- [13] Kaggle, "H&M Personalized Fashion Recommendations — Evaluation." [Online]. Available: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/overview/evaluation — V
- [14] C. Deotte, "How to build a GBT ranker model," OTTO recommender system competition, Kaggle discussion. [Online]. Available: https://www.kaggle.com/competitions/otto-recommender-system/discussion/370210 — V
- [15] T. Van de Wiele, "Santander product recommendation competition, 2nd place winner's solution write-up," Kaggle Blog (No Free Hunch), Jan. 12, 2017. [Online]. Available: https://web.archive.org/web/20170115012200id_/http://blog.kaggle.com:80/2017/01/12/santander-product-recommendation-competition-2nd-place-winners-solution-write-up-tom-van-de-wiele/ — V; две детали исправлены (C, §9.4)
- [16] E. Chen, "Instacart market basket analysis, winner's interview: 2nd place, Kazuki Onodera," Kaggle Blog (No Free Hunch), Sep. 21, 2017. [Online]. Available: https://web.archive.org/web/20170922014019id_/http://blog.kaggle.com/2017/09/21/instacart-market-basket-analysis-winners-interview-2nd-place-kazuki-onodera/ — V; одна атрибуция исправлена (C)
- [17] Wp-Zhang, "H-M-Fashion-RecSys," GitHub repository. [Online]. Available: https://github.com/Wp-Zhang/H-M-Fashion-RecSys — V; метрику MAP@12 брать из [13]
- [18] Kaggle Team, "Profiling top Kagglers: Bestfitting, currently #1 in the world," Kaggle Blog (No Free Hunch), May 7, 2018. [Online]. Available: https://web.archive.org/web/20180515173145id_/http://blog.kaggle.com/2018/05/07/profiling-top-kagglers-bestfitting-currently-1-in-the-world/ — V
- [19] G. Park, "The dangers of overfitting: A Kaggle postmortem," Jul. 6, 2012. [Online]. Available: https://gregpark.io/blog/Kaggle-Psychopathy-Postmortem — V
- [20] K. Onodera, T. Viel, G. Titericz, and C. Deotte, "The Kaggle Grandmasters playbook: 7 battle-tested modeling techniques for tabular data," NVIDIA Technical Blog, Sep. 18, 2025. [Online]. Available: https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/ — V
- [21] T. Viel, "Build efficient recommender systems with co-visitation matrices and NVIDIA cuDF," NVIDIA Technical Blog, Aug. 21, 2024. [Online]. Available: https://developer.nvidia.com/blog/build-efficient-recommender-systems-with-co-visitation-matrices-and-rapids-cudf/ — V (дата и автор). Заголовок взят из трека, а URL содержит «rapids-cudf» — сверить на странице.
- [22] C. Deotte, "Pro tips for building multilingual recommender systems," NVIDIA Technical Blog, Aug. 10, 2023. [Online]. Available: https://developer.nvidia.com/blog/pro-tips-for-building-multilingual-recommender-systems/ — V
- [23] C. Deotte, "Grandmaster pro tip: Winning first place in Kaggle competition with feature engineering using NVIDIA cuDF pandas," NVIDIA Technical Blog, Apr. 17, 2025. [Online]. Available: https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas/ — V
- [24] C. Deotte, "Grandmaster pro tip: Winning first place in a Kaggle competition with stacking using cuML," NVIDIA Technical Blog, May 22, 2025. [Online]. Available: https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/ — V
- [25] C. Deotte, "Winning a Kaggle competition with generative AI-assisted coding," NVIDIA Technical Blog, Apr. 23, 2026. [Online]. Available: https://developer.nvidia.com/blog/winning-a-kaggle-competition-with-generative-ai-assisted-coding/ — V
- [26] MLWave, "Kaggle ensembling guide," Jun. 11, 2015. [Online]. Available: https://web.archive.org/web/20160206102007id_/http://mlwave.com/kaggle-ensembling-guide/ — V
- [27] Z. Zając, "Adversarial validation, part one," FastML, May 23, 2016. [Online]. Available: http://fastml.com/adversarial-validation-part-one/ — V
- [28] PacktPublishing, "The-Kaggle-Book," GitHub repository (README; chapter_06/adversarial-validation-example.ipynb). [Online]. Available: https://raw.githubusercontent.com/PacktPublishing/The-Kaggle-Book/main/README.md ; https://raw.githubusercontent.com/PacktPublishing/The-Kaggle-Book/main/chapter_06/adversarial-validation-example.ipynb — V
- [29] ML Contests, "The state of machine learning competitions (2024 edition)." [Online]. Available: https://mlcontests.com/state-of-machine-learning-competitions-2024/ — C
- [30] HSE University / Coursera, "How to win a data science competition: Learn from top Kagglers," course page, archived 2020. [Online]. Available: https://web.archive.org/web/20200416102958id_/https://www.coursera.org/learn/competitive-data-science — V
- [31] otto-de, "recsys-dataset: KAGGLE.md," GitHub. [Online]. Available: https://github.com/otto-de/recsys-dataset/blob/main/KAGGLE.md — V
- [32] benhamner, "Metrics: ml_metrics/average_precision.py," GitHub. [Online]. Available: https://raw.githubusercontent.com/benhamner/Metrics/master/Python/ml_metrics/average_precision.py — V
- [33] Implicit 0.7.2 documentation, "AlternatingLeastSquares." [Online]. Available: https://benfred.github.io/implicit/api/models/cpu/als.html — V
- [34] LightGBM documentation, "Parameters," and source file "src/metric/map_metric.hpp." [Online]. Available: https://lightgbm.readthedocs.io/en/latest/Parameters.html ; https://raw.githubusercontent.com/microsoft/LightGBM/master/src/metric/map_metric.hpp — V
- [35] XGBoost documentation, "Learning to rank." [Online]. Available: https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html — V
- [36] Kaggle, "sweet.tv – TV program recommender," competition pages (Data, Evaluation, Description). [Online]. Available: https://www.kaggle.com/competitions/sweettv-tv-program-recommender — веб-страница отдаёт пустую JS-оболочку; содержимое сверено по дампу API [L1] (V/C).

### 9.3 Локальные файлы
- [L1] /Users/konstantin/Documents/вшэ/recsys_course/midterm/evidence/kaggle_meta/pages.txt
- [L2] /Users/konstantin/Documents/вшэ/recsys_course/midterm/evidence/kaggle_meta/topics_leaderboard.txt ; /Users/konstantin/Documents/вшэ/recsys_course/midterm/evidence/kaggle_meta/public_leaderboard_2020_final.csv
- [L3] /Users/konstantin/Documents/вшэ/recsys_course/midterm/evidence/kaggle_submissions.csv
- [L4] /Users/konstantin/Documents/вшэ/recsys_course/midterm/evidence/eda.json
- [L5] /Users/konstantin/Documents/вшэ/recsys_course/midterm/evidence/prep.json
- [L6] /Users/konstantin/Documents/вшэ/recsys_course/midterm/evidence/tiebreak_sensitivity_F1.json
- [L7] /Users/konstantin/Documents/вшэ/recsys_course/midterm/evidence/tz_alignment.json
- [L8] /Users/konstantin/Documents/вшэ/recsys_course/midterm/evidence/candidates.json
- [L9] /Users/konstantin/Documents/вшэ/recsys_course/midterm/evidence/baselines.json
- [L10] /Users/konstantin/Documents/вшэ/recsys_course/midterm/evidence/methods_F1.json
- [L11] /Users/konstantin/Documents/вшэ/recsys_course/midterm/evidence/als_sweep_F1.json
- [L12] /Users/konstantin/Documents/вшэ/recsys_course/midterm/evidence/hybrid_ablation_F1.json
- [L13] /Users/konstantin/Documents/вшэ/recsys_course/midterm/evidence/experiments_F1.json
- [L14] /Users/konstantin/Documents/вшэ/recsys_course/midterm/evidence/experiments2_F1.json
- [L15] /Users/konstantin/Documents/вшэ/recsys_course/midterm/evidence/experiments3_F1.json. «Чистый» пользователь — тот, у кого в цели есть сессии хотя бы в 11 из 12 недель; определение в /Users/konstantin/Documents/вшэ/recsys_course/midterm/src/experiments3.py, строка 36.
- [L16] /Users/konstantin/Documents/вшэ/recsys_course/midterm/evidence/ranker_v1.json
- [L17] /Users/konstantin/Documents/вшэ/recsys_course/midterm/evidence/sub02_hybrid_lr31_r250_s5_info.json ; /Users/konstantin/Documents/вшэ/recsys_course/midterm/evidence/sub01_ranker_v1_info.json
- [L18] /Users/konstantin/Documents/вшэ/recsys_course/midterm/README.md
- [L19] /Users/konstantin/Documents/вшэ/recsys_course/midterm/tests/check_submission.py ; /Users/konstantin/Documents/вшэ/recsys_course/midterm/evidence/reproduce_sub01.json
- [L20] /Users/konstantin/Documents/вшэ/recsys_course/midterm/src/metrics.py ; /Users/konstantin/Documents/вшэ/recsys_course/midterm/src/data.py (тай-брейк id_asc, строки 71–83)

### 9.4 Отклонено или цитируется только в исправленном виде
- **The Kaggle Book** (Banachewicz & Massaron, Packt, 2022). Год и название не подтверждены: Packt и O'Reilly отдают 403, в Open Library только предрелизная запись 2021 года под названием «Data Analysis and Machine Learning with Kaggle». Цитировать только репозиторий [28].
- **Bestfitting**, цитаты «Find similar Kaggle competitions» и «prepare the solution document in the very beginning» — проверщик их не видел, его вход был обрезан. Не цитировать.
- **Greg Park**, цитата «I chose my five final submissions based on cross-validation estimates…» — среди проверенных её нет. Не цитировать; проверена, например, «did not include my best submissions in the final five».
- **Исправленные формулировки** — использовать только в исправленном виде:
  - ML Contests 2024: LightGBM — самая частая GBDT-библиотека (16 решений против 13 у CatBoost и 8 у XGBoost). Самая частая библиотека вообще — PyTorch: 53 из 60 решений с глубоким обучением, всего найдено 79 решений.
  - Santander 2nd, MAP-оптимизация: «great effects in local validation (~0.2% boost) but limited value on the public leaderboard»; на private эффект был так же положительным.
  - Santander 2nd, результат 0.031: получен примерно за 90 минут на самых важных признаках, по одной модели на пару product-lag. «Top-10 features» — отдельный совет автора.
  - Instacart: «how ready the user is to repurchase» относится к признаку Days_last_order-max.
  - H&M: MAP@12 в README Wp-Zhang не назван — ссылаться на [13].
  - H&M 2nd, цитата про availability: она из общего раздела команды, а не лично Jankiewicz.
  - Roelofs: порог 10 000 примеров выведен для соревнований на accuracy.
  - Bestfitting: в источнике «a period of time», а не «a later period».
  - sweet.tv: меток не «до 5», а строго 5; деление 50/50 — официальное правило, а не допущение; Kaggle-этап закончился 21.11.2020 в 23:59, а не 22 ноября; горизонт F1 равен 12 неделям только номинально — внутри провал 25–27.

### 9.5 Статус проверки не получен (проверка обрезана) — не цитировать до проверки
- OTTO 3rd (Deotte): https://www.kaggle.com/competitions/otto-recommender-system/writeups/g-b-d-t-3rd-place-using-only-rules-achieves-lb-0-5
- OTTO 1st (mrkmakr): https://www.kaggle.com/competitions/otto-recommender-system/writeups/mrkmakr-1st-place-solution
- OTTO 2nd: https://www.kaggle.com/competitions/otto-recommender-system/discussion/382839 ; часть ONODERA: https://www.kaggle.com/competitions/otto-recommender-system/writeups/sos3-2nd-place-solution-onodera-part
- OTTO, Radek Osmulski: https://www.kaggle.com/competitions/otto-recommender-system/discussion/364991
- Santander:
  - 1-е место, idle_speculation: https://www.kaggle.com/competitions/santander-product-recommendation/writeups/idle-speculation-1-solution
  - 3-е место, Jack (Japan): https://www.kaggle.com/competitions/santander-product-recommendation/writeups/jack-japan-3rd-place-solution-with-code
  - BreakfastPirate: https://www.kaggle.com/competitions/santander-product-recommendation/discussion/26786
  - 2-е место, Kaggle-writeup: https://www.kaggle.com/competitions/santander-product-recommendation/writeups/tom-van-de-wiele-2nd-place-solution — вместо него использовать блог [15].
- Instacart: снимок Wayback 20170929051359 (вместо него использовать [16]); README репозитория https://github.com/KazukiOnodera/Instacart. Утверждения «total_buy_n5 — важнейший признак», «order_ratio_by_chance», «добавил в обучение 3 последних заказа» не проверены.
- Rekko Challenge, 2-е место: https://habr.com/ru/post/454818/ ; 14-е место: https://habr.com/ru/post/461055/
- logicai-io/recsys2019: https://github.com/logicai-io/recsys2019
- LightGBM: утверждения «lambdarank_truncation_level ≈ k + 3» и «rank_xendcg achieves the similar performance as lambdarank» среди проверенных цитат [34] отсутствуют.

### 9.6 Пробел
Проверенных ссылок на базовые работы по методам курса — ALS для неявной обратной связи, ассоциативные правила, PageRank/PPR, content-based filtering — в этом входе нет: трек «литература» не пришёл. Брать их только после его проверки и не цитировать по памяти.