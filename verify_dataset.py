"""
# ==============================================================================
# VERİ SETİ DOĞRULAMA ARACI — Eksik/Bozuk Kayıtları Bulmak İçin Kod
# ==============================================================================
"""

import os
import numpy as np

DATA_PATH      = "MP_Data"
NO_SEQUENCES   = 30
SEQ_LENGTH     = 30
FEATURE_DIM    = 1662

# ── Klasörleri otomatik olarak tarayıp alfabe sırasına diziyoruz
actions = sorted([
    d for d in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, d))
])

print("=" * 62)
print("  VERİ SETİ DOĞRULAMA RAPORU")
print("=" * 62)
print(f"  Bulunan jest sınıfları ({len(actions)} adet): {actions}\n")

total_ok      = 0
total_missing = 0
total_corrupt = 0
report_lines  = []

for action in actions:
    ok = missing = corrupt = 0
    for seq in range(NO_SEQUENCES):
        for frame in range(SEQ_LENGTH):
            path = os.path.join(DATA_PATH, action, str(seq), f"{frame}.npy")

            # Dosya var mı? Kontrol et
            if not os.path.exists(path):
                missing += 1
                continue

            # Boyut doğru mu? kontrol et
            try:
                arr = np.load(path)
                if arr.shape != (FEATURE_DIM,):
                    corrupt += 1
            except Exception:
                corrupt += 1
                continue

            ok += 1

    total_ok      += ok
    total_missing += missing
    total_corrupt += corrupt

    status = "✓ TAM" if (missing == 0 and corrupt == 0) else "⚠ EKSİK/BOZUK"
    line = (f"  {action:<25} "
            f"ok={ok:>4}  missing={missing:>3}  corrupt={corrupt:>3}  [{status}]")
    print(line)
    report_lines.append(line)

print("\n" + "-" * 62)
print(f"  TOPLAM  ok={total_ok}  missing={total_missing}  corrupt={total_corrupt}")

if total_missing == 0 and total_corrupt == 0:
    print("\n  ✅ Veri seti temiz. Eğitime geçebilirsin.")
    print(f"\n  train_model.py dosyasında şu satırı güncelle:")
    print(f"\n  actions = np.array({actions})\n")
else:
    print("\n  ❌ Sorunlu dosyalar var. Eksik jestleri yeniden kaydet.")

print("=" * 62)