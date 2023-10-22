from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from nltk.tokenize import sent_tokenize


class Tranlationmanytoenglish():
    def __init__(self, device='cpu', max_length=512):
        self.model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-one-mmt").to(device)
        self.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
        self.language_to_id = {
            'Arabic': 'ar_AR', 'Czech': 'cs_CZ', 'German': 'de_DE', 'English': 'en_XX',
            'Spanish': 'es_XX', 'Estonian': 'et_EE', 'Finnish': 'fi_FI', 'French': 'fr_XX',
            'Gujarati': 'gu_IN', 'Hindi': 'hi_IN', 'Italian': 'it_IT', 'Japanese': 'ja_XX',
            'Kazakh': 'kk_KZ', 'Korean': 'ko_KR', 'Lithuanian': 'lt_LT', 'Latvian': 'lv_LV',
            'Burmese': 'my_MM', 'Nepali': 'ne_NP', 'Dutch': 'nl_XX', 'Romanian': 'ro_RO',
            'Russian': 'ru_RU', 'Sinhala': 'si_LK', 'Turkish': 'tr_TR', 'Vietnamese': 'vi_VN',
            'Chinese': 'zh_CN', 'Afrikaans': 'af_ZA', 'Azerbaijani': 'az_AZ', 'Bengali': 'bn_IN',
            'Persian': 'fa_IR', 'Hebrew': 'he_IL', 'Croatian': 'hr_HR', 'Indonesian': 'id_ID',
            'Georgian': 'ka_GE', 'Khmer': 'km_KH', 'Macedonian': 'mk_MK', 'Malayalam': 'ml_IN',
            'Mongolian': 'mn_MN', 'Marathi': 'mr_IN', 'Polish': 'pl_PL', 'Pashto': 'ps_AF',
            'Portuguese': 'pt_XX', 'Swedish': 'sv_SE', 'Swahili': 'sw_KE', 'Tamil': 'ta_IN',
            'Telugu': 'te_IN', 'Thai': 'th_TH', 'Tagalog': 'tl_XX', 'Ukrainian': 'uk_UA',
            'Urdu': 'ur_PK', 'Xhosa': 'xh_ZA', 'Galician': 'gl_ES', 'Slovene': 'sl_SI'
        }
        self.device = device
        self.max_length = max_length

    def __call__(self, doc, language):
        language_token = self.language_to_id[language]
        self.tokenizer.src_lang = language_token
        doc_chunks = self.create_document_chunks(doc)
        encoded_text = self.tokenizer(doc_chunks, return_tensors='pt', padding=True).to(self.device)
        generated_tokens = self.model.generate(**encoded_text)
        translated_text_chunks = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        translated_text = " ".join(translated_text_chunks)
        return translated_text

    def create_document_chunks(self, text):
        """
        To create chunks of text with a maximum number of tokens
        """
        sentences = sent_tokenize(text)
        current_chunk = ""
        chunks = []
        curr_chunk_length = 0
        for sentence in sentences:
            current_sentence_tokens_length = len(self.tokenizer(current_chunk)['input_ids'])
            if curr_chunk_length + current_sentence_tokens_length >= self.max_length:
                chunks.append(current_chunk)
                curr_chunk_length = current_sentence_tokens_length
                current_chunk = sentence
            else:
                current_chunk += " " + sentence
                curr_chunk_length += current_sentence_tokens_length

        # Add the last chunk
        if current_chunk != "":
            chunks.append(current_chunk)

        return chunks
