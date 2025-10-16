from __future__ import annotations
import re
from langdetect import detect, DetectorFactory



def extract_sql_only(text: str) -> str:
    """
    LLM bazen açıklama/markdown döndürüyor. Bu fonksiyon sadece çalıştırılabilir
    SQL'i bırakır. ```sql ... ``` varsa içini, yoksa tüm metni strip eder.
    """
    if not text:
        return ""
    m = re.search(r"```sql\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip().rstrip(";")
    # Kod bloğu yoksa yine de sadeleştir
    # Bazı modeller başa/sona metin ekliyor; ilk SELECT/WITH'ten kırp
    m2 = re.search(r"\b(WITH|SELECT)\b(.+)$", text, flags=re.IGNORECASE | re.DOTALL)
    return (m2.group(0) if m2 else text).strip().rstrip(";")


def _fallback_detect(s: str) -> str:
    """
    Basit ve bağımlılıksız dil kestirimi:
    Türkçe karakter (ç,ğ,ı,ö,ş,ü) görürse 'tr', aksi halde 'en'.
    """
    if not s or not s.strip():
        return "tr"
    t = s.lower()
    if any(ch in t for ch in "çğıöşü"):
        return "tr"
    return "en"

def detect_lang(s: str) -> str:
    """
    'tr' veya 'en' döndürür.
    Önce langdetect varsa onu dener; yoksa basit fallback'e düşer.
    """
    try:
        # Bu import'u fonksiyon içine alıyoruz ki modül yüklenirken
        # döngüsel import veya paket eksikliği sorun yaratmasın.
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        code = detect(s or "")
        if code and code.lower().startswith("tr"):
            return "tr"
        return "en"
    except Exception:
        return _fallback_detect(s)

__all__ = ["extract_sql_only", "detect_lang"]