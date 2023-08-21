from datasets import load_dataset

PROMPT_TEMPLATE = {
    "summarization": [
        ("""Đoạn văn:\n{document}.\n\nTóm tắt đoạn văn trên:\n"""),
        (
            "<s> [INST] <<SYS>>\n"
            "Nhiệm vụ của bạn là tóm tắt đoạn văn bản sau, đưa ra câu trả lời là bản tóm tắt:\n"
            "<</SYS>>\n"
            "```{document}``` "
            "[/INST]"
        ),
        (
            "<s> [INST] <<SYS>>\n"
            "Bạn là một trợ lý hữu dụng, biết tôn trọng và thành thật. Bạn luôn luôn trả lời các câu hỏi một cách có ích nhiều nhất có thể, "
            "nhưng đồng thời phải an toàn. "
            "Câu trả lời của bạn không được bao gồm các ngôn từ độc hại, phân biệt chủng tộc, phân biệt giới tính, nguy hiểm, nội dung vi phạm pháp luật. "
            "Nhiệm vụ của bạn là tóm tắt đoạn văn bản nằm trong triple backtick. Bài tóm tắt phải đầy đủ các thông tin quan trọng, ngắn gọn và thu hút người đọc. "
            "Ngôn ngữ bạn phải sử dụng để tóm tắt là tiếng Việt.\n"
            "<</SYS>>\n"
            "```{document}``` "
            "[/INST]"
        ),
    ],
    "question-answering": [
        ("""Ngữ cảnh:\n{context}.\n\nCâu hỏi:\n{question}.\n\nTrả lời:\n"""),
        (
            "<s> [INST] <<SYS>>\n"
            "Hãy trả lời câu hỏi bên dưới bằng tiếng Việt "
            "với các thông tin được cung cấp trong phần ngữ cảnh. "
            "Nếu trong ngữ cảnh không có đủ thông tin, "
            'hãy trả lời "Tôi không biết".\n'
            "<</SYS>>\n"
            """Ngữ cảnh: {context}\n"""
            """Câu hỏi: {question}\n"""
            "Trả lời: "
            "[/INST]"
        ),
        (
            "<s> [INST] <<SYS>>\n"
            "Bạn là một trợ lý hữu dụng sử dụng tiếng Việt, biết tôn trọng và thành thật. Bạn luôn luôn trả lời các câu hỏi một cách có ích nhiều nhất có thể, "
            "nhưng đồng thời phải an toàn. "
            "Câu trả lời của bạn không được bao gồm các ngôn từ độc hại, phân biệt chủng tộc, phân biệt giới tính, nguy hiểm, nội dung vi phạm pháp luật. "
            "Làm ơn hãy chắc chắn câu trả lời của bạn tự nhiên, tích cực và không thiên vị bất cứ cái gì. "
            "Nếu có câu hỏi không hợp lý hoặc không rõ ràng thì hãy giải thích tại sao thay vì trả lời không đúng sự thật. "
            "Nếu bạn không biết câu trả lời thì đừng chia sẻ thông tin sai sự thật.\n"
            "<</SYS>>\n"
            """Nhiệm vụ của bạn là dựa vào đoạn văn nằm trong dấu triple backtick, hãy trả lời câu hỏi sau bằng tiếng Việt: {question}\n"""
            """Đoạn văn: ```{context}``` """
            "[/INST]"
        ),
    ],
    "translation": [
        ("""Câu hỏi:\nDịch "{document}" sang tiếng Anh.\n\nTrả lời:\n"""),
    ],
    "text-generation": [
        """{context} """,
    ],
}

def eval_answers(sample):
    sample['answers'] = eval(sample["answers"])
    return sample

class DatasetWrapper:
    def __init__(self, dataset_name, prompting_strategy=0) -> None:
        self.dataset_name = dataset_name
        self.prompting_strategy = prompting_strategy

        self.get_dataset_config()
        self.get_prompt()

    def get_prompt(self):
        if self.prompting_strategy not in [0, 1, 2]:
            raise ValueError("Prompting strategy is not supported")

        self.prompt = PROMPT_TEMPLATE[self.task][self.prompting_strategy]

    def get_dataset_config(self):
        if self.dataset_name == "Yuhthe/vietnews":
            self.task = "summarization"
            self.dataset = load_dataset(self.dataset_name, split="test")
            self.dataset.set_format(columns=["article", "abstract"])
            self.original_text = "article"
            self.summarized_text = "abstract"
            
        elif self.dataset_name == "vietnews_robustness":
            self.task = "summarization"
            self.dataset = load_dataset('csv', data_files="evaluation_datasets/vietnews_for_robustness.csv", split="train")
            self.dataset.set_format(columns=["article", "abstract"])
            self.original_text = "article"
            self.summarized_text = "abstract"

        elif self.dataset_name == "GEM/wiki_lingua":
            self.task = "summarization"
            self.dataset = load_dataset(self.dataset_name, "vi", split="test")
            self.original_text = "source"
            self.summarized_text = "target"
            
        elif self.dataset_name == "wiki_lingua_robustness":
            self.task = "summarization"
            self.dataset = load_dataset('csv', data_files="evaluation_datasets/wiki_lingua_for_robustness.csv", split="train")
            self.original_text = "source"
            self.summarized_text = "target"

        elif self.dataset_name == "VIMQA":
            self.task = "question-answering"
            pass

        elif self.dataset_name == "juletxara/xquad_xtreme":
            self.task = "question-answering"
            self.dataset = load_dataset(self.dataset_name, "vi", split="test")
            self.context = "context"
            self.question = "question"
            self.answer = "answers"
            
        elif self.dataset_name == "xquad_xtreme_robustness":
            self.task = "question-answering"
            self.dataset = load_dataset('csv', data_files="evaluation_datasets/xquad_xtreme_for_robustness.csv", split="train")
            self.dataset = self.dataset.map(eval_answers)
            self.context = "context"
            self.question = "question"
            self.answer = "answers"

        elif self.dataset_name == "mlqa":
            self.task = "question-answering"
            self.dataset = load_dataset(self.dataset_name, "mlqa.vi.vi", split="test")
            self.context = "context"
            self.question = "question"
            self.answer = "answers"
        
        elif self.dataset_name == "mlqa_robustness":
            self.task = "question-answering"
            self.dataset = load_dataset('csv', data_files="evaluation_datasets/mlqa_for_robustness.csv", split="train")
            self.dataset = self.dataset.map(eval_answers)
            self.context = "context"
            self.question = "question"
            self.answer = "answers"

        elif self.dataset_name == "vietgpt/opus100_envi":
            self.task = "translation"
            self.dataset = load_dataset(self.dataset_name, split="test")
            self.source_language = "vi"
            self.target_language = "en"

        elif self.dataset_name == "mt_eng_vietnamese":
            self.task = "translation"
            self.dataset = load_dataset(
                self.dataset_name, "iwslt2015-vi-en", split="test"
            )
            self.source_language = "vi"
            self.target_language = "en"

        elif self.dataset_name == "vietgpt/wikipedia_vi":
            self.task = "text-generation"
            self.dataset = load_dataset(self.dataset_name, split="test")

        else:
            raise ValueError("Dataset is not supported")

    def get_dataset(self):
        return self.dataset
