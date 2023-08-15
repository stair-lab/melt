from datasets import load_dataset

class DatasetWrapper:
    def __init__(self, dataset_name) -> None:
        self.dataset_name = dataset_name
        self.get_dataset_config()
    
    def get_dataset_config(self):
        if self.dataset_name == "Yuhthe/vietnews":
            self.task = "summarization"
            self.dataset = load_dataset(self.dataset_name, split='test')
            self.original_text = "article"
            self.summarized_text = "abstract"
            self.prompt = """Đoạn văn:\n{}.\n\nTóm tắt đoạn văn trên:\n"""
            
        elif self.dataset_name == "VIMQA":
            self.task = "question-answering"
            pass
        
        elif self.dataset_name == "juletxara/xquad_xtreme":
            self.task = "question-answering"
            self.dataset = load_dataset(self.dataset_name,'vi', split='test')
            self.context = "context"
            self.question = "question"
            self.answer = "answers"
            self.prompt = """Ngữ cảnh:\n{}.\n\nCâu hỏi:\n{}.\n\nTrả lời:\n"""
            
        elif self.dataset_name == "vietgpt/opus100_envi":
            self.task = "translation"
            self.dataset = load_dataset(self.dataset_name, split='test')
            self.source_language = "vi"
            self.target_language = "en"
            self.prompt = '''Câu hỏi:\nDịch "{}" sang tiếng Anh.\n\nTrả lời:\n'''
            
        elif self.dataset_name == "mt_eng_vietnamese":
            self.task = "translation"
            self.dataset = load_dataset(self.dataset_name,'iwslt2015-vi-en', split='test')
            self.source_language = "vi"
            self.target_language = "en"
            
        elif self.dataset_name == "vietgpt/wikipedia_vi":
            self.task = "text-generation"
            self.dataset = load_dataset(self.dataset_name, split='test')
            
        else:
            raise ValueError("Dataset is not supported")
        
    def get_dataset(self):
        return self.dataset