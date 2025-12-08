# qwen2.5-codegen-lora

Qwen2.5-Coder LoRA Fine-Tuning on CodeGen-Deep-5K & CodeGen-Diverse-5K

Bu proje, Qwen/Qwen2.5-Coder-1.5B-Instruct modelinin
CodeGen-Deep-5K ve CodeGen-Diverse-5K datasetleri üzerinde
QLoRA yöntemi ile fine-tune edilmesini, checkpoint değerlendirmelerini ve
en iyi model seçimini içerir.

Proje Amacı

Kod üretim modelleri için derin reasoning ve farklı problem kapsama alanı sunan iki dataset üzerinde ince ayarlı (LoRA) eğitim gerçekleştirmek
Eğitim sürecinde oluşan birden fazla checkpoint’i değerlendirip en iyi performans veren modeli seçmek
Tüm pipeline’ı GitHub & HuggingFace üzerinde dökümleyerek araştırma/proje teslim hedeflerini karşılamak

Dataset Açıklaması

CodeGen-Deep-5K
1000 farklı problem × her biri için 5 farklı çözüm
Reasoning + final çözüm içerir
Derin düşünme gerektiren problemlerde modelin adım adım mantık kurmasını öğretir
CodeGen-Diverse-5K
5000 benzersiz problem
Her problem için 1 çözüm
Geniş problem çeşitliliği → modelin farklı kodlama stillerine uyum sağlamasını öğretir
Her iki dataset’in solution alanı eğitimde kullanılmıştır.

Eğitim Pipeline’ı (QLoRA)

Eğitimler Colab GPU üzerinde, aşağıdaki adımlarla yapılmıştır:
Base model 4-bit quantization ile yüklendi.
LoRA katmanları eklendi. (rank=64, alpha=16, dropout=0.1)
Dataset’in solution alanı tokenize edildi.

TrainingArguments:

logging_steps = 50
eval_steps = 100
save_steps = 100
num_train_epochs = 1
gradient_accumulation_steps = 4
learning_rate = 2e-4
Eğitim sırasında her 100 adımda checkpoint oluşturuldu
Eğitim sonunda LoRA adapter’ı qwen2.5-*-lora-ckpt100 klasörüne kaydedildi

Kullanılan Hyperparameter’lar

LoRA Parametreleri
Parametre	Değer
Rank (r)	64
Alpha	16
Dropout	0.1
Target Modules	q_proj, k_proj, v_proj, o_proj
Bias	none
Training Parametreleri
Parametre	Değer
Learning Rate	2e-4
Batch Size	1
Gradient Accumulation	4
Effective Batch Size	4
Epoch	1
Max Seq Length	2048
Optimizer	paged_adamw_8bit
Scheduler	cosine
Warmup Ratio	0.03
Logging Steps	50
Eval Steps	100
Checkpoint Save Steps	100

Training Scripts

Aşağıdaki iki script GitHub içinde scripts/ klasöründedir:
🔹 train_deep.py
Deep dataset ile eğitir
Model + tokenizer + LoRA yükler
Checkpoint’leri her 100 adımda kaydeder
Final LoRA adapter’ını qwen2.5-deep-lora-ckpt100/ içine yazar
🔹 train_diverse.py
Diverse dataset ile eğitir
Aynı pipeline, farklı dataset
Sonuçlar qwen2.5-diverse-lora-ckpt100/ içine kaydedilir
Her iki script, Colab notebook’taki ile birebir aynı davranışı gösterir.

Evaluation Script (Checkpoint Seçimi)

Script yolu:
scripts/eval_checkpoints.py

Yaptıkları:
Deep & Diverse için tüm checkpoint klasörlerini okur
İlk 100 örnekten oluşan test split ile evaluation yapar
Her checkpoint için eval loss hesaplar
En iyi checkpoint’i otomatik seçer
Sonuçları checkpoint_summary.json dosyasına kaydeder

Checkpoint Selection Sonuçları
Dataset	En iyi checkpoint	Eval Loss
DEEP	checkpoint-1250	≈ 0.4231
DIVERSE	checkpoint-1100	≈ 0.4734

Test Split Politikası

Test datası asla eğitimde kullanılmaz
Hem DEEP hem DIVERSE için ilk 100 örnek test seti olarak ayrıldı
Checkpoint evaluation işlemi bu test split üzerinde yapıldı
Bu, Görev 4’ün gerekliliklerine birebir uygundur.


HuggingFace Modelleri (yüklendikten sonra)

Model linkleri buraya eklenecek:
🔹 Deep LoRA Model
https://huggingface.co/<username>/qwen2.5-deep-lora
🔹 Diverse LoRA Model
https://huggingface.co/<username>/qwen2.5-diverse-lora

Eğitim Logları

Tüm eğitim logları GitHub’da logs/ klasöründedir:

logs/deep_train.log
logs/diverse_train.log
logs/deep_eval.log
logs/diverse_eval.log


Bu loglar:
Train loss → her 50 step
Eval loss → her 100 step
Checkpoint oluşumu → her 100 step
bilgilerini içerir.

Sonuç

Bu proje:
QLoRA ile iki farklı dataset üzerinde başarılı ince ayar
Loss eğrileri stabil ve anlamlı
Checkpoint selection süreciyle optimum performans garantisi
Modeller HuggingFace’e aktarılabilir.
Scriptler ve tüm pipeline GitHub’da reproducible şekilde yer alır.

Dosya Yapısı

qwen2.5-codegen-lora/
├─ scripts/
│  ├─ train_deep.py
│  ├─ train_diverse.py
│  └─ eval_checkpoints.py
├─ logs/
│  ├─ deep_train.log
│  ├─ diverse_train.log
│  ├─ deep_eval.log
│  ├─ diverse_eval.log
├─ models/
├─ notebook/
│  └─ Lora.ipynb
├─ checkpoint_summary.json
├─ requirements.txt
└─ README.md
