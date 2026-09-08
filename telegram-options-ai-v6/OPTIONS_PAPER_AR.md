# تشغيل عقود SPY + SPX على البوت — بدون Alpaca Key

## طبقة AI الاختيارية

الإصدار 6 يدعم Chronos-2 + TimesFM 2.5 + FinBERT كـEnsemble اختياري فوق الإشارة الفنية. استخدم `/optionsmodels` لمعرفة الحالة، وراجع [MODELS_AR.md](MODELS_AR.md) للتثبيت. لا يتغير كون جميع العقود Paper ونظرية.


هذه النسخة **Paper فقط**. لا تشتري عقدًا حقيقيًا ولا تتصل بوسيط. تستخدم سعر SPY/SPX من مصدر بيانات عام بدون مفتاح، ثم تنشئ عقدًا نظريًا داخليًا وتحسب Premium وDelta وP/L للمحاكاة.

## التشغيل

بعد إعداد Telegram مرة واحدة وتشغيل الخادم:

```bash
python -m telegram_bridge setup
python -m telegram_bridge serve
```

على Windows استخدم مثلًا:

```powershell
.\.venv\Scripts\python.exe -m telegram_bridge setup
.\.venv\Scripts\python.exe -m telegram_bridge serve
```

لا تحتاج `configure-trading` ولا Alpaca Key لتشغيل محاكاة العقود.

## أوامر Telegram

```text
/optionson
/optionsstatus
/optionpositions
/optiontrades
/optionpnl
/optionsoff
```

- `/optionson`: يبدأ المحرك.
- `/optionsoff`: يوقف فتح/معالجة دورات جديدة.
- `/optionsstatus`: حالة المحرك وPaper equity.
- `/optionpositions`: المراكز المفتوحة.
- `/optiontrades`: آخر الصفقات المغلقة.
- `/optionpnl`: الربح والخسارة التجريبي.

## ماذا يفعل تلقائيًا؟

كل 5 دقائق أثناء الجلسة العادية يفحص SPY وSPX في نفس الدورة. يستخدم EMA قصير/بطيء مع حركة آخر نحو 30 دقيقة. إذا اتفق الاتجاه يختار Call للصعود أو Put للهبوط؛ وإذا كانت الإشارة مختلطة ينتظر.

عند قوة إشارة عالية وقبل 2:00 مساءً بتوقيت نيويورك يختار 0DTE، وإلا ينتقل إلى 1DTE/يوم التداول التالي. يبني Strikes نظرية حول سعر الأصل ويختار الأقرب إلى |Delta| ≈ 0.35.

- SPX: نموذج European-style وتسوية نقدية في المحاكاة.
- SPY: نموذج American-style وتسوية أسهم في المحاكاة.
- لا يحدث Exercise أو تسليم فعلي لأن النظام Paper فقط.

## إدارة محفظة Paper

- رصيد بداية افتراضي: 100,000 USD.
- أقصى ميزانية دخول للصفقة: 5% من Paper equity.
- عقود كاملة فقط، multiplier = 100.
- مركز واحد فقط لكل أصل في نفس الوقت.
- Target افتراضي: +35% من Premium.
- Stop افتراضي: -25% من Premium.
- حد خسارة يومي محقق: 3% من رصيد البداية؛ بعده لا يفتح صفقات جديدة في ذلك اليوم.
- يمنع تكرار قرار نفس الأصل في نفس نافذة الخمس دقائق حتى بعد restart.

## مهم

Premium وGreeks والـOption Chain كلها **نظرية/مصطنعة** وليست Bid/Ask حقيقية من سوق الخيارات. مصدر سعر الأصل بدون مفتاح قد يتغير أو يتوقف؛ عند غياب بيانات حديثة لا يفتح المحرك صفقة. هذه النسخة هدفها اختبار المنطق والسلوك قبل ربط مزود Option Chain ووسيط حقيقي لاحقًا.