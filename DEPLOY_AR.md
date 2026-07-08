# نشر خادم ImplantScan على Railway

هذا الخادم يسوي المطابقة الحقيقية (ICP بـ open3d) — نفس اللي نجح بالاختبار على الماركرات الثلاثة.

## الملفات
- `main.py` — الخادم (FastAPI + open3d)
- `requirements.txt` — المكتبات
- `Dockerfile` — بيئة التشغيل (تتضمن مكتبات النظام اللي يحتاجها open3d)
- `railway.json` — إعدادات Railway

## خطوات النشر

### الطريقة الأسهل — عبر GitHub
1. ارفع هذي الملفات الأربعة على ريبو GitHub (تقدر تستخدم ريبو `Implant-scan-` الموجود عندك، أو ريبو جديد).
2. بحساب Railway: New Project → Deploy from GitHub repo → اختر الريبو.
3. Railway يكتشف الـ Dockerfile تلقائياً ويبني الخادم.
4. بعد ما يخلص، روح Settings → Networking → Generate Domain، وتطلع لك رابط عام (مثلاً `https://implant-scan-production.up.railway.app`).

### التأكد إنه شغّال
افتح الرابط بالمتصفح — لازم يطلع:
```json
{"status":"ok","service":"ImplantScan ICP"}
```

## ربط الواجهة بالخادم
1. افتح `implantscan.html`.
2. اضغط زر **«الخادم»** فوق.
3. الصق رابط Railway (مثلاً `https://implant-scan-production.up.railway.app`).
4. خلاص — من الحين أي نقرة على ماركر تروح للخادم وترجع بموضع ودوران دقيق.

> لو تركت حقل الخادم فارغ، البرنامج يستخدم المطابقة المحلية بالمتصفح (أضعف — للتجربة السريعة فقط).

## ملاحظات
- أول طلب بعد النشر ممكن يكون بطيء شوي (الخادم "يصحى"). الطلبات بعدها أسرع.
- البناء أول مرة ياخذ عدة دقائق (open3d مكتبة كبيرة).
- لو صار خطأ بالبناء، تأكد إن Railway يستخدم الـ Dockerfile (مو Nixpacks) — إعدادات `railway.json` تفرض هذا.
