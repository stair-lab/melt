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
