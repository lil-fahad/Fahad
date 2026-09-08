# التداول التلقائي التجريبي — Alpaca Paper

هذه النسخة تضيف شراء وبيع تلقائي للأسهم الأمريكية على **حساب Alpaca Paper فقط**. المحرك يستخدم نموذج التوقع الموجود أصلًا، لكنه لا ينفذ لمجرد وجود توقع: يجب أن يجتاز التقرير فحوص الجودة والحداثة، وأن تكون جلسة السوق العادية مفتوحة عند الإرسال، وأن يكون اتصال Telegram الخاص بالمالك سليمًا.

> التداول الحقيقي التلقائي غير مدعوم في هذه النسخة. الأوامر اليدوية القديمة ما زالت تحتاج `/confirm`.

## تشغيل Windows

من PowerShell داخل مجلد `telegram-bridge`:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m telegram_bridge setup
.\.venv\Scripts\python.exe -m telegram_bridge configure-trading --mode paper
```

بعد إعداد Telegram ومفاتيح حساب Alpaca Paper، اختر الأسهم والحدود التجريبية:

```powershell
.\.venv\Scripts\python.exe -m telegram_bridge configure-autotrading --symbols AAPL,MSFT --order-usd 100 --max-position-usd 500
.\.venv\Scripts\python.exe -m telegram_bridge doctor
.\.venv\Scripts\python.exe -m telegram_bridge serve
```

الأسماء والمبالغ أعلاه مثال تقني فقط وليست توصية استثمارية.

في محادثة Telegram الخاصة المحددة أثناء الإعداد:

```text
/autostatus
/autoon
```

## التحكم

| الأمر | الوظيفة |
|---|---|
| `/autoon` | يبدأ المحرك التلقائي التجريبي وفق السياسة المحفوظة |
| `/autooff` | يوقف أي إرسال تلقائي جديد |
| `/autostatus` | يعرض الحالة وآخر قرار وسببه |
| `/pause` | يوقف جميع عمليات التداول الجديدة، اليدوية والتلقائية |
| `/resume` | يرفع الإيقاف العام |
| `/account` | يعرض الحساب والمراكز |
| `/orders` | يصالح حالات الأوامر مع الوسيط |
| `/order TICKET` | يعرض أمرًا محددًا |
| `/cancel TICKET` | يطلب إلغاء أمر قابل للإلغاء |

من الجهاز نفسه:

```powershell
.\.venv\Scripts\python.exe -m telegram_bridge autotrading-status
.\.venv\Scripts\python.exe -m telegram_bridge halt-autotrading
```

## كيف يقرر؟

الافتراضي:

- شراء: تقرير صالح، إشارة صعود، `probability_up >= 0.60`، والعائد المتوقع موجب.
- بيع: تقرير صالح، إشارة هبوط، `probability_up <= 0.40`، والعائد المتوقع سالب.
- انتظار: إذا امتنع النموذج أو فشل أي شرط جودة/حداثة/سوق/رصيد/ملكية/سعر.

فحوص التقرير تشمل: مصدر Alpaca IEX، بيانات split-adjusted غير اصطناعية، آخر جلسة مكتملة، 40 فترة تقييم على الأقل، ونتيجة Brier وMAE أفضل من خط الأساس مع تغطية نطاق لا تقل عن 70%.

## حدود التنفيذ

- حساب Paper فقط، ويُرفض التشغيل التلقائي في live mode.
- أسهم كاملة فقط؛ لا short ولا اقتراض ولا pyramiding تلقائي.
- الشراء يلتزم `--order-usd` و`--max-position-usd` وحدود التداول الأصلية والنقد المتاح.
- البيع يقتصر على الأسهم التي يثبت سجل المحرك أنه اشتراها ونُفذت فعليًا، مع مطابقة مركز الوسيط الحالي.
- عرض IEX يجب أن يكون حديثًا وصالحًا؛ السبريد التلقائي الأقصى 1%.
- قرار واحد لكل سهم لكل جلسة سوق مكتملة، محفوظ في SQLite.
- إذا كانت نتيجة POST غير محسومة، يستخدم نفس `client_order_id` للمصالحة ولا يكرر الإرسال.
- إذا توقف `/autooff` أو `/pause` أو فقد البرنامج poller lease أثناء الفحوص، يعيد التحقق **بعد preflight وقبل POST** ويرفض الإرسال.
- لا يوجد stop-loss أو take-profit أو تصفية نهاية اليوم في هذه النسخة.

## تغيير السياسة

أوقف `serve` قبل تعديل الإعداد. إعادة `configure-autotrading` تحفظ السياسة في وضع **متوقف**، ثم أعد تشغيل `serve` وأرسل `/autoon` من محادثة المالك. تغيير الحساب أو الرموز أو الحدود أو الاحتمالات يغيّر بصمة السياسة ويمنع تشغيل حالة قديمة بالخطأ.

خيارات الإعداد:

```text
--symbols AAPL,MSFT
--order-usd 100
--max-position-usd 500
--horizon 1|5|20
--interval-seconds 60..3600
--buy-probability 0.60
--sell-probability 0.40
--no-notify
```

## البيانات والاسترداد

احتفظ بـ`private/config.json` وملف SQLite داخل `data_dir`. جدول `auto_decisions` وسجل `trades` ضروريان لمنع التكرار وتتبع ملكية الاستراتيجية. لا تنشئ قاعدة فارغة بدل القديمة عند النقل.

الإشعار التلقائي للمالك مستقل عن صلاحية MCP العامة للإرسال. أداة MCP الخاصة بالحالة **قراءة فقط** ولا تستطيع تشغيل المحرك أو تنفيذ صفقة.

## التحقق

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.lock
.\.venv\Scripts\python.exe -m pytest -q
```

اختبارات التطوير تستخدم HTTP محاكى ولا ترسل أي طلب حقيقي إلى Telegram أو Alpaca. راجع `VALIDATION.md` لنتيجة هذه الحزمة.
