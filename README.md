# 🎌 AutoScanlate AR

**مترجم كوميكس ومانغا تلقائي — إنجليزي → عربي**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/menoali/Autoscanlate-ar2/blob/main/AutoScanlate_AR.ipynb)

## المميزات
- 🤖 Claude Vision يكشف الفقاعات تلقائياً
- 🗑 Contour Detection يمسح شكل الفقاعة كاملاً بما فيه الذيل
- 📦 دعم CBZ / CBR / ZIP
- ⚡ حتى 100 صفحة دفعة واحدة
- 📱 يشتغل من الآيفون عبر Google Colab

## الاستخدام السريع (Colab)
1. افتح [AutoScanlate_AR.ipynb](AutoScanlate_AR.ipynb) في Colab
2. أدخل مفتاح Anthropic API
3. ارفع الملف وشغّل

## الاستخدام على الكمبيوتر
```bash
pip install -r requirements.txt
python autoscanlate.py --input chapter.cbz --api_key sk-ant-...
```

## المتطلبات
- Python 3.8+
- مفتاح Anthropic API من [console.anthropic.com](https://console.anthropic.com)