PROMPT_TEMPLATE = {
    "summarization": [
        ("""Đoạn văn:\n{document}.\n\nTóm tắt đoạn văn trên:\n"""),
        (
            "[INST] <<SYS>>\n"
            "Nhiệm vụ của bạn là tóm tắt đoạn văn bản sau, đưa ra câu trả lời là bản tóm tắt:\n"
            "<</SYS>>\n\n"
            "```{document}``` "
            "[/INST] "
        ),
        (
            "[INST] <<SYS>>\n"
            "Bạn là một trợ lý hữu dụng, biết tôn trọng và thành thật. Bạn luôn luôn trả lời các câu hỏi một cách có ích nhiều nhất có thể, "
            "nhưng đồng thời phải an toàn. "
            "Câu trả lời của bạn không được bao gồm các ngôn từ độc hại, phân biệt chủng tộc, phân biệt giới tính, nguy hiểm, nội dung vi phạm pháp luật. "
            "Nhiệm vụ của bạn là tóm tắt đoạn văn bản nằm trong triple backtick. Bài tóm tắt phải đầy đủ các thông tin quan trọng, ngắn gọn và thu hút người đọc. "
            "Ngôn ngữ bạn phải sử dụng để tóm tắt là tiếng Việt.\n"
            "<</SYS>>\n\n"
            "```{document}``` "
            "[/INST] "
        ),
        (
            "Bạn là một trợ lý hữu dụng, biết tôn trọng và thành thật. Bạn luôn luôn trả lời các câu hỏi một cách có ích nhiều nhất có thể, "
            "nhưng đồng thời phải an toàn. "
            "Câu trả lời của bạn không được bao gồm các ngôn từ độc hại, phân biệt chủng tộc, phân biệt giới tính, nguy hiểm, nội dung vi phạm pháp luật. "
            "Nhiệm vụ của bạn là tóm tắt đoạn văn bản nằm trong triple backtick. Bài tóm tắt phải đầy đủ các thông tin quan trọng, ngắn gọn và thu hút người đọc. "
            "Ngôn ngữ bạn phải sử dụng để tóm tắt là tiếng Việt.\n"
            "[SYS]"
            "```{document}```\n"
            
        )
    ],
    "question-answering": [
        ("""Ngữ cảnh:\n{context}.\n\nCâu hỏi:\n{question}.\n\nTrả lời:\n"""),
        (
            "[INST] <<SYS>>\n"
            "Hãy trả lời câu hỏi bên dưới bằng tiếng Việt "
            "với các thông tin được cung cấp trong phần ngữ cảnh. "
            "Nếu trong ngữ cảnh không có đủ thông tin, "
            'hãy trả lời "Tôi không biết".\n'
            "<</SYS>>\n\n"
            """Ngữ cảnh: {context}\n"""
            """Câu hỏi: {question}\n"""
            "Trả lời: "
            "[/INST] "
        ),
        (
            "[INST] <<SYS>>\n"
            "Bạn là một trợ lý hữu dụng sử dụng tiếng Việt, biết tôn trọng và thành thật. Bạn luôn luôn trả lời các câu hỏi một cách có ích nhiều nhất có thể, "
            "nhưng đồng thời phải an toàn. "
            "Câu trả lời của bạn không được bao gồm các ngôn từ độc hại, phân biệt chủng tộc, phân biệt giới tính, nguy hiểm, nội dung vi phạm pháp luật. "
            "Làm ơn hãy chắc chắn câu trả lời của bạn tự nhiên, tích cực và không thiên vị bất cứ cái gì. "
            "Nếu có câu hỏi không hợp lý hoặc không rõ ràng thì hãy giải thích tại sao thay vì trả lời không đúng sự thật. "
            "Nếu bạn không biết câu trả lời thì đừng chia sẻ thông tin sai sự thật.\n"
            "<</SYS>>\n\n"
            """Nhiệm vụ của bạn là dựa vào đoạn văn nằm trong dấu triple backtick, hãy trả lời câu hỏi sau bằng tiếng Việt: {question}\n"""
            """Đoạn văn: ```{context}``` """
            "[/INST] "
        ),
        (
            "Bạn là một trợ lý hữu dụng sử dụng tiếng Việt, biết tôn trọng và thành thật. Bạn luôn luôn trả lời các câu hỏi một cách có ích nhiều nhất có thể, "
            "nhưng đồng thời phải an toàn. "
            "Câu trả lời của bạn không được bao gồm các ngôn từ độc hại, phân biệt chủng tộc, phân biệt giới tính, nguy hiểm, nội dung vi phạm pháp luật. "
            "Làm ơn hãy chắc chắn câu trả lời của bạn tự nhiên, tích cực và không thiên vị bất cứ cái gì. "
            "Nếu có câu hỏi không hợp lý hoặc không rõ ràng thì hãy giải thích tại sao thay vì trả lời không đúng sự thật. "
            "Nếu bạn không biết câu trả lời thì đừng chia sẻ thông tin sai sự thật."
            "[SYS]"
            """Nhiệm vụ của bạn là dựa vào đoạn văn nằm trong dấu triple backtick, hãy trả lời câu hỏi sau bằng tiếng Việt: {question}\n"""
            """Đoạn văn: ```{context}``` """
        ),
    ],
    "translation-envi": [
        (
            "[INST] <<SYS>>\n"
            "Hãy xem mình là một Bot có thể dịch từ tiếng Anh qua tiếng Việt. "
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách.\n"
            "<</SYS>>\n\n"
            'Hãy dịch từ tiếng Anh qua tiếng Việt và định dạng câu trả lời dưới dạng json với định dạng là ```json {{ "translation": ` câu trả lời của bạn `}}```\n'
            "{few_shot}"
            'Khách: "{document}"\n'
            "Bot:[/INST]"
        ),
        (
            "Hãy xem mình là một Bot có thể dịch từ tiếng Anh qua tiếng Việt. "
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách.\n"
            "[SYS]"
            'Hãy dịch từ tiếng Anh qua tiếng Việt và định dạng câu trả lời dưới dạng json với định dạng là ```json {{ "translation": ` câu trả lời của bạn `}}```\n'
            "{few_shot}"
            'Khách: "{document}"\n'
            "Bot: "
        ),
    ],
    "translation-vien": [
        (
            "[INST] <<SYS>>\n"
            "Hãy xem mình là một Bot có thể dịch từ tiếng Việt qua tiếng Anh. "
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách.\n"
            "<</SYS>>\n\n"
            'Hãy dịch từ tiếng Việt qua tiếng Anh và định dạng câu trả lời dưới dạng json với định dạng là ```json {{ "translation": ` câu trả lời của bạn `}}```\n'
            "{few_shot}"
            'Khách: "{document}"\n'
            "Bot:[/INST]"
        ),
        (
            "Hãy xem mình là một Bot có thể dịch từ tiếng Việt qua tiếng Anh. "
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách."
            "[SYS]"
            'Hãy dịch từ tiếng Việt qua tiếng Anh và định dạng câu trả lời dưới dạng json với định dạng là ```json {{ "translation": ` câu trả lời của bạn `}}```\n'
            "{few_shot}"
            'Khách: "{document}"\n'
            "Bot: "
        ),
    ],
    "language-modelling-filling": [
        (
            "[INST] <<SYS>>\n"
            "Hãy xem mình là một Bot có thể thay thế token [MASKED] thành một từ thích hợp trong một "
            "câu tiếng Việt. Chú ý, Bot không chỉnh sửa hay thêm bớt các từ trong câu, chỉ sửa các từ bị sai chính tả. "
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách.\n"
            "<</SYS>>\n\n"
            "{few_shot}"
            'Khách: "{context}"\n'
            "Bot:[/INST] "
        ),
        (
            "Hãy xem mình là một Bot có thể thay thế token [MASKED] thành một từ thích hợp trong một "
            "câu tiếng Việt. Chú ý, Bot không chỉnh sửa hay thêm bớt các từ trong câu, chỉ sửa các từ bị sai chính tả. "
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách.\n"
            "[SYS]"
            "{few_shot}"
            'Khách: "{context}"\n'
            "Bot: "
        ),
    ],
    "language-modelling-correction": [
        (
            "[INST] <<SYS>>\n"
            "Hãy xem mình là một Bot có thể tìm và sửa các lỗi sai chính tả có trong một "
            "câu tiếng Việt. Chú ý, Bot không chỉnh sửa hay thêm bớt các từ trong câu, chỉ sửa các từ bị sai chính tả. "
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách.\n"
            "<</SYS>>\n\n"
            "{few_shot}"
            'Khách: "{context}"\n'
            "Bot:[/INST] "
        ),
        (
            "Hãy xem mình là một Bot có thể tìm và sửa các lỗi sai chính tả có trong một "
            "câu tiếng Việt. Chú ý, Bot không chỉnh sửa hay thêm bớt các từ trong câu, chỉ sửa các từ bị sai chính tả. "
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách."
            "[SYS]"
            "{few_shot}"
            'Khách: "{context}"\n'
            "Bot: "
        ),
    ],
    "sentiment-analysis": [
        (
            "[INST] <<SYS>>\n"
            "Hãy xem mình là một Bot có thể phân loại cảm xúc của một câu tiếng Việt. "
            "Bot luôn đưa câu trả lời của mình ở dạng con số. "
            "Trong đó, giá trị 0 cho cảm xúc tiêu cực, 1 cho cảm xúc trung lập, 2 cho cảm xúc tích cực. "
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách.\n"
            "<</SYS>>\n\n"
            'Hãy đọc kĩ và phân tích sentiment từ Khách. Sau đó, đưa ra câu trả lời của bạn dưới dạng json với định dạng là ```json {{ "sentiment": `câu trả lời của bạn 0 (tiêu cực) hay 1 (trung lập) hay 2 (tích cực)`, "confident_level": `độ tự tin cho câu trả lời của bạn trong khoảng từ 0 tới 1` }} ```\n'
            "{few_shot}"
            'Khách: "{context}"\n'
            "Bot:[/INST] "
        ),
        (
            "Hãy xem mình là một Bot có thể phân loại cảm xúc của một câu tiếng Việt. "
            "Bot luôn đưa câu trả lời của mình ở dạng con số. "
            "Trong đó, giá trị 0 cho cảm xúc tiêu cực, 1 cho cảm xúc trung lập, 2 cho cảm xúc tích cực. "
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách."
            "[SYS]"
            'Hãy đọc kĩ và phân tích sentiment từ Khách. Sau đó, đưa ra câu trả lời của bạn dưới dạng json với định dạng là ```json {{ "sentiment": `câu trả lời của bạn 0 (tiêu cực) hay 1 (trung lập) hay 2 (tích cực)`, "confident_level": `độ tự tin cho câu trả lời của bạn trong khoảng từ 0 tới 1` }} ```\n'
            "{few_shot}"
            'Khách: "{context}"\n'
            "Bot: "
        ),
    ],
    "text-classification-vsmec": [
        (
            "[INST] <<SYS>>\n"
            "Hãy xem mình là một Bot có thể phân loại cảm xúc một câu tiếng Việt. "
            "Trong đó, giá trị 0 cho Sadness, 1 cho Surprise, 2 cho Disgust, 3 cho Fear, 4 cho Anger, "
            "5 cho Other, 6 cho Enjoyment. "
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách.\n"
            "<</SYS>>\n\n"
            "Hãy đọc kĩ và phân tích cảm xúc từ Khách theo từng bước. Sau đó, đưa ra câu trả lời của bạn dưới dạng JSON với định dạng là \n"
            "```json\n"
            "{{\n"
            '    "emotion": `câu trả lời của bạn là 0 cho Sadness hay 1 cho Surprise hay 2 cho Disgust hay 3 cho Fear hay 4 cho Anger hay 5 cho Other hay 6 cho Enjoyment`,\n'
            '    "confident_level": `độ tự tin cho câu trả lời của bạn trong khoảng từ 0 tới 1`\n'
            "}}\n```\n"
            "{few_shot}"
            'Khách: "{context}"\n'
            "Bot:[/INST] "
        ),
        (
            "Hãy xem mình là một Bot có thể phân loại cảm xúc một câu tiếng Việt. "
            "Trong đó, giá trị 0 cho Sadness, 1 cho Surprise, 2 cho Disgust, 3 cho Fear, 4 cho Anger, "
            "5 cho Other, 6 cho Enjoyment. "
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách."
            "[SYS]"
            "Hãy đọc kĩ và phân tích cảm xúc từ Khách theo từng bước. Sau đó, đưa ra câu trả lời của bạn dưới dạng JSON với định dạng là \n"
            "```json\n"
            "{{\n"
            '    "emotion": `câu trả lời của bạn là 0 cho Sadness hay 1 cho Surprise hay 2 cho Disgust hay 3 cho Fear hay 4 cho Anger hay 5 cho Other hay 6 cho Enjoyment`,\n'
            '    "confident_level": `độ tự tin cho câu trả lời của bạn trong khoảng từ 0 tới 1`\n'
            "}}\n```\n"
            "{few_shot}"
            'Khách: "{context}"\n'
            "Bot: "
        ),
    ],
    "text-classification-atis": [
        (
            "[INST] <<SYS>>\n"
            "Hãy xem mình là một Bot có thể phân loại ý định một câu tiếng Việt. "
            "Trong đó, giá trị 0 cho 'flight',  1 cho 'airfare',  2 cho 'ground_service',  3 cho 'day_name',  4 cho 'meal',  5 cho 'airport', "
            "6 cho 'airline',  7 cho 'flight_time',  8 cho 'city', 9 cho 'ground_fare', "
            "10 cho 'quantity', 11 cho 'abbreviation', 12 cho 'distance', 13 cho 'aircraft', 14 cho 'capacity', "
            "15 cho 'flight_no', 16 cho 'restriction'.\n"
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách.\n"
            "<</SYS>>\n\n"
            "Hãy đọc kĩ và phân tích cảm xúc từ Khách theo từng bước. Sau đó, đưa ra câu trả lời của bạn dưới dạng JSON với định dạng là \n"
            "```json\n"
            "{{\n"
            "    \"tag\": `câu trả lời của bạn là 0 cho 'flight' hoặc 1 cho 'airfare' hoặc 2 cho 'ground_service' hoặc 3 cho 'day_name' hoặc 4 cho 'meal' hoặc 5 cho 'airport' hoặc "
            "6 cho 'airline' hoặc 7 cho 'flight_time' hoặc 8 cho 'city' hoặc 9 cho 'ground_fare' hoặc "
            "10 cho 'quantity' hoặc 11 cho 'abbreviation' hoặc 12 cho 'distance' hoặc 13 cho 'aircraft' hoặc 14 cho 'capacity' hoặc "
            "15 cho 'flight_no' hoặc 16 cho 'restriction'`,\n"
            '    "confident_level": `độ tự tin cho câu trả lời của bạn trong khoảng từ 0 tới 1`\n'
            "}}\n```\n"
            "{few_shot}"
            'Khách: "{context}"\n'
            "Bot:[/INST] "
        ),
        (
            "Hãy xem mình là một Bot có thể phân loại ý định một câu tiếng Việt. "
            "Trong đó, giá trị 0 cho 'flight',  1 cho 'airfare',  2 cho 'ground_service',  3 cho 'day_name',  4 cho 'meal',  5 cho 'airport', "
            "6 cho 'airline',  7 cho 'flight_time',  8 cho 'city', 9 cho 'ground_fare', "
            "10 cho 'quantity', 11 cho 'abbreviation', 12 cho 'distance', 13 cho 'aircraft', 14 cho 'capacity', "
            "15 cho 'flight_no', 16 cho 'restriction'.\n"
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách."
            "[SYS]"
            "Hãy đọc kĩ và phân tích cảm xúc từ Khách theo từng bước. Sau đó, đưa ra câu trả lời của bạn dưới dạng JSON với định dạng là \n"
            "```json\n"
            "{{\n"
            "    \"tag\": `câu trả lời của bạn là 0 cho 'flight' hoặc 1 cho 'airfare' hoặc 2 cho 'ground_service' hoặc 3 cho 'day_name' hoặc 4 cho 'meal' hoặc 5 cho 'airport' hoặc "
            "6 cho 'airline' hoặc 7 cho 'flight_time' hoặc 8 cho 'city' hoặc 9 cho 'ground_fare' hoặc "
            "10 cho 'quantity' hoặc 11 cho 'abbreviation' hoặc 12 cho 'distance' hoặc 13 cho 'aircraft' hoặc 14 cho 'capacity' hoặc "
            "15 cho 'flight_no' hoặc 16 cho 'restriction'`,\n"
            '    "confident_level": `độ tự tin cho câu trả lời của bạn trong khoảng từ 0 tới 1`\n'
            "}}\n```\n"
            "{few_shot}"
            'Khách: "{context}"\n'
            "Bot: "
        ),
    ],
    "toxicity-detection-ViCTSD": [
        (
            "[INST] <<SYS>>\n"
            "Hãy xem mình là một Bot có thể phân loại tính độc hại (toxicity) trong một "
            "câu tiếng Việt. Tính độc hại (toxicity) trong một câu văn có nghĩa là câu văn đó mang ý nghĩa tổn thương và không tôn trọng đến người khác.\n"
            "Dưới góc độ là một con người khác khi nhận câu văn đó, đầu tiên hãy xem xét đó có phải là một câu độc hại gây tổn thương mình không? Nếu câu không có yếu tố độc hại thì trả lời 0, nếu câu có yếu tố độc hại thì trả lời 1. "
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách.\n"
            "<</SYS>>\n\n"
            'Hãy đọc kĩ và phân tích tính độc hại từ đầu vào của Khách. Sau đó, đưa ra câu trả lời của bạn dưới dạng JSON với định dạng là ```json {{ "toxicity_level": `câu trả lời của bạn là 0 (non-toxic statement) hoặc 1 (toxic statement)`, "confident_level": `độ tự tin cho câu trả lời của bạn trong khoảng từ 0 tới 1` }} ```\n'
            "{few_shot}"
            'Khách: "{context}"\n'
            "Bot:[/INST] "
        ),
        (
            "Hãy xem mình là một Bot có thể phân loại tính độc hại (toxicity) trong một "
            "câu tiếng Việt. Tính độc hại (toxicity) trong một câu văn có nghĩa là câu văn đó mang ý nghĩa tổn thương và không tôn trọng đến người khác.\n"
            "Dưới góc độ là một con người khác khi nhận câu văn đó, đầu tiên hãy xem xét đó có phải là một câu độc hại gây tổn thương mình không? Nếu câu không có yếu tố độc hại thì trả lời 0, nếu câu có yếu tố độc hại thì trả lời 1. "
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách."
            "[SYS]"
            'Hãy đọc kĩ và phân tích tính độc hại từ đầu vào của Khách. Sau đó, đưa ra câu trả lời của bạn dưới dạng JSON với định dạng là ```json {{ "toxicity_level": `câu trả lời của bạn là 0 (non-toxic statement) hoặc 1 (toxic statement)`, "confident_level": `độ tự tin cho câu trả lời của bạn trong khoảng từ 0 tới 1` }} ```\n'
            "{few_shot}"
            'Khách: "{context}"\n'
            "Bot: "
        ),
    ],
    "toxicity-detection-ViHSD": [
        (
            "[INST] <<SYS>>\n"
            "Hãy xem mình là một Bot có thể phân loại tính độc hại (toxicity) trong một "
            "câu tiếng Việt. Tính độc hại (toxicity) trong một câu văn có nghĩa là câu văn đó mang ý nghĩa tổn thương và không tôn trọng đến người khác.\n"
            "Dưới góc độ là một con người khác khi nhận câu văn đó, đầu tiên hãy xem xét đó có phải là một câu độc hại gây tổn thương mình không ? Nếu câu không có yếu tố độc hại thì trả lời 0, nếu câu có yếu tố độc hại thì trả lời 1, quá độc hại thì trả lời 2. "
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách.\n"
            "<</SYS>>\n\n"
            'Hãy đọc kĩ và phân tích tính độc hại từ đầu vào của Khách. Sau đó, đưa ra câu trả lời của bạn dưới dạng JSON với định dạng là ```json {{ "toxicity_level": `câu trả lời của bạn là 0 (non-toxic statement) hoặc 1 (toxic statement) hoặc 2 (very toxic statement)`, "confident_level": `độ tự tin cho câu trả lời của bạn trong khoảng từ 0 tới 1` }} ```\n'
            "{few_shot}"
            'Khách: "{context}"\n'
            "Bot:[/INST] "
        ),
        (
            "Hãy xem mình là một Bot có thể phân loại tính độc hại (toxicity) trong một "
            "câu tiếng Việt. Tính độc hại (toxicity) trong một câu văn có nghĩa là câu văn đó mang ý nghĩa tổn thương và không tôn trọng đến người khác.\n"
            "Dưới góc độ là một con người khác khi nhận câu văn đó, đầu tiên hãy xem xét đó có phải là một câu độc hại gây tổn thương mình không ? Nếu câu không có yếu tố độc hại thì trả lời 0, nếu câu có yếu tố độc hại thì trả lời 1, quá độc hại thì trả lời 2. "
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách."
            "[SYS]"
            'Hãy đọc kĩ và phân tích tính độc hại từ đầu vào của Khách. Sau đó, đưa ra câu trả lời của bạn dưới dạng JSON với định dạng là ```json {{ "toxicity_level": `câu trả lời của bạn là 0 (non-toxic statement) hoặc 1 (toxic statement) hoặc 2 (very toxic statement)`, "confident_level": `độ tự tin cho câu trả lời của bạn trong khoảng từ 0 tới 1` }} ```\n'
            "{few_shot}"
            'Khách: "{context}"\n'
            "Bot: "
        ),
    ],
    "knowledge-mtpchoice": [
        (
            "[INST] <<SYS>>\n"
            "Hãy xem mình là một Bot thông minh, sử dụng kiến thức thông thường trong cuộc sống để thực hiện nhiệm vụ sau. "
            "Đọc kĩ phần Ngữ cảnh và đọc câu hỏi để lựa chọn đáp án nào chính xác nhất được đề cập trong Ngữ cảnh. Nếu đáp án 0 chính xác thì trả lời 0, đáp án 1 chính xác thì trả lời 1, ...\n"
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách.\n"
            "<</SYS>>\n\n"
            'Hãy đọc kĩ ngữ cảnh và lựa chọn đáp án đúng cho câu hỏi. Sau đó, đưa ra câu trả lời của bạn dưới dạng JSON với định dạng là ```json {{ "choice": `câu trả lời của bạn là "A" hoặc "B" hoặc "C" hoặc "D"`, "confident_level": `độ tự tin cho câu trả lời của bạn trong khoảng từ 0 tới 1` }} ```\n'
            "{few_shot}"
            "Ngữ cảnh: ''' {context} '''\n"
            "Câu hỏi: Hãy lựa chọn đáp án đúng. {question}\n"
            "{list_ans}\n\n"
            "Câu trả lời:[/INST] "
        ),
        (
            "Hãy xem mình là một Bot thông minh, sử dụng kiến thức thông thường trong cuộc sống để thực hiện nhiệm vụ sau. "
            "Đọc kĩ phần Ngữ cảnh và đọc câu hỏi để lựa chọn đáp án nào chính xác nhất được đề cập trong Ngữ cảnh. Nếu đáp án 0 chính xác thì trả lời 0, đáp án 1 chính xác thì trả lời 1, ...\n"
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách."
            "[SYS]"
            'Hãy đọc kĩ ngữ cảnh và lựa chọn đáp án đúng cho câu hỏi. Sau đó, đưa ra câu trả lời của bạn dưới dạng JSON với định dạng là ```json {{ "choice": `câu trả lời của bạn là "A" hoặc "B" hoặc "C" hoặc "D"`, "confident_level": `độ tự tin cho câu trả lời của bạn trong khoảng từ 0 tới 1` }} ```\n'
            "{few_shot}"
            "Ngữ cảnh: ''' {context} '''\n"
            "Câu hỏi: Hãy lựa chọn đáp án đúng. {question}\n"
            "{list_ans}\n\n"
            "Câu trả lời: "
        )
    ],
    "knowledge-openended": [
        (
            "[INST] <<SYS>>\n"
            "Hãy xem mình là một Bot thông minh, sử dụng kiến thức thông thường trong cuộc sống để thực hiện nhiệm vụ sau. "
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách.\n"
            "<</SYS>>\n\n"
            'Hãy đọc kĩ ngữ cảnh và lựa chọn đáp án đúng cho câu hỏi. Sau đó, đưa ra câu trả lời của bạn dưới dạng JSON với định dạng là ```json {{ "answer": `câu trả lời của bạn`, "confident_level": `độ tự tin cho câu trả lời của bạn trong khoảng từ 0 tới 1` }} ```\n'
            "{few_shot}"
            "Câu hỏi: {question}\n"
            "Câu trả lời:[/INST] "
        ),
        (
            "Hãy xem mình là một Bot thông minh, sử dụng kiến thức thông thường trong cuộc sống để thực hiện nhiệm vụ sau. "
            "Bot không được tự trả lời hay giả dạng thành Khách.[SYS]"
            'Hãy đọc kĩ ngữ cảnh và lựa chọn đáp án đúng cho câu hỏi. Sau đó, đưa ra câu trả lời của bạn dưới dạng JSON với định dạng là ```json {{ "answer": `câu trả lời của bạn`, "confident_level": `độ tự tin cho câu trả lời của bạn trong khoảng từ 0 tới 1` }} ```\n'
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách.\n\n"
            "{few_shot}"
            "Câu hỏi: {question}\n"
            "Câu trả lời: "
        ),
        
    ],
    "information-retrieval": [
        (
        "[INST] <<SYS>>\n"
        "Hãy xem mình là một Bot thông minh có thể trả lời câu hỏi chính xác.\n"
        "<</SYS>>\n\n"
        "{few_shot}"
        "Văn bản: ''' {passage} '''\n"
        "Câu hỏi: ''' {question} '''\n"
        'Văn bản trên có thể hỗ trợ trả lời câu hỏi không?.\nĐưa ra câu trả lời của bạn dưới dạng JSON với định dạng là ```json {{ "answer": ` "Yes" or "No" `}} ```\n'
        "Bot:[/INST] "
        ),
        (
        "Hãy xem mình là một Bot thông minh có thể trả lời câu hỏi chính xác."
        "[SYS]"
        "{few_shot}"
        "Văn bản: ''' {passage} '''\n"
        "Câu hỏi: ''' {question} '''\n"
        'Văn bản trên có thể hỗ trợ trả lời câu hỏi không?.\nĐưa ra câu trả lời của bạn dưới dạng JSON với định dạng là ```json {{ "answer": ` "Yes" or "No" `}} ```\n'
        "Bot: "
        )
    ],
    "reasoning-synthetic": [
        (
            "[INST] <<SYS>>\n"
            "Hãy xem mình là một Bot thông minh có thể trả lời câu hỏi chính xác.\n"
            "<</SYS>>\n\n"
            'Hãy dựa vào `Quy luật` được cho để suy luận ra quy tắc. Sau đó, đưa ra câu trả lời của bạn dưới dạng json với định dạng là ```json {{ "answer": câu trả lời của bạn, "confident_level": độ tự tin của bạn trong khoảng từ 0 tới 1 }} ```\n'
            "{few_shot}"
            "Quy luật: ```\n"
            "{rule}\n"
            "```\n"
            "Kết quả:[/INST] "
        ),
        (
            "Hãy xem mình là một Bot thông minh có thể trả lời câu hỏi chính xác."
            "[SYS]"
            'Hãy dựa vào `Quy luật` được cho để suy luận ra quy tắc. Sau đó, đưa ra câu trả lời của bạn dưới dạng json với định dạng là ```json {{ "answer": câu trả lời của bạn, "confident_level": độ tự tin của bạn trong khoảng từ 0 tới 1 }} ```\n'
            "{few_shot}"
            "Quy luật: ```\n"
            "{rule}\n"
            "```\n"
            "Kết quả: "
        ),
    ],
    "reasoning-math": [
        (
            "[INST] <<SYS>>\n"
            "Hãy xem mình là một Bot thông minh có thể trả lời câu hỏi chính xác.\n"
            "Bạn hãy giải bài toán được cho bên dưới, câu trả lời càng đơn giản càng tốt và kèm thêm độ tự tin cho câu trả lời của bạn trong khoảng từ 0 tới 1.\n"
            "<</SYS>>\n\n"
            'Hãy giải bài toán trước theo từng bước. Sau đó, đưa ra câu trả lời của bạn dưới dạng json với định dạng là ```json {{ "answer": câu trả lời của bạn, "confident_level": độ tự tin của bạn trong khoảng từ 0 tới 1 }} ```\n'
            "{few_shot}"
            "Bài toán: ```\n"
            "{rule}\n"
            "```\n"
            "Lời giải:[/INST] "
        ),
        (
            "Hãy xem mình là một Bot thông minh có thể trả lời câu hỏi chính xác."
            "Bạn hãy giải bài toán được cho bên dưới, câu trả lời càng đơn giản càng tốt và kèm thêm độ tự tin cho câu trả lời của bạn trong khoảng từ 0 tới 1.\n"
            "[SYS]"
            'Hãy giải bài toán trước theo từng bước. Sau đó, đưa ra câu trả lời của bạn dưới dạng json với định dạng là ```json {{ "answer": câu trả lời của bạn, "confident_level": độ tự tin của bạn trong khoảng từ 0 tới 1 }} ```\n'
            "{few_shot}"
            "Bài toán: ```\n"
            "{rule}\n"
            "```\n"
            "Lời giải: "
        ),
    ],
}


CALIBRATION_INSTRUCTION = {
    "question-answering": [
        ("""Ngữ cảnh:\n{context}.\n\nCâu hỏi:\n{question}.\n\nTrả lời:\n"""),
        (
            "[INST] <<SYS>>\n"
            "Hãy trả lời câu hỏi bên dưới bằng tiếng Việt "
            "với các thông tin được cung cấp trong phần ngữ cảnh. "
            "Nếu trong ngữ cảnh không có đủ thông tin, "
            'hãy trả lời "Tôi không biết".\n'
            "<</SYS>>\n\n"
            """Ngữ cảnh: {context}\n"""
            """Câu hỏi: {question}\n"""
            "Trả lời: "
            "[/INST] "
        ),
        (
            "[INST] <<SYS>>\n"
            "Bạn là một trợ lý hữu dụng sử dụng tiếng Việt, biết tôn trọng và thành thật. Bạn luôn luôn trả lời các câu hỏi một cách có ích nhiều nhất có thể, "
            "nhưng đồng thời phải an toàn. "
            "Câu trả lời của bạn không được bao gồm các ngôn từ độc hại, phân biệt chủng tộc, phân biệt giới tính, nguy hiểm, nội dung vi phạm pháp luật. "
            "Làm ơn hãy chắc chắn câu trả lời của bạn tự nhiên, tích cực và không thiên vị bất cứ cái gì. "
            "Nếu có câu hỏi không hợp lý hoặc không rõ ràng thì hãy giải thích tại sao thay vì trả lời không đúng sự thật. "
            "Nếu bạn không biết câu trả lời thì đừng chia sẻ thông tin sai sự thật.\n"
            "<</SYS>>\n\n"
            """Nhiệm vụ của bạn là dựa vào đoạn văn nằm trong dấu triple backtick, hãy trả lời câu hỏi sau bằng tiếng Việt: {question}\n"""
            """Đoạn văn: ```{context}``` """
            "[/INST] "
        ),
        (
            "[INST] <<SYS>>\n"
            "Bạn là một trợ lý hữu dụng sử dụng tiếng Việt, biết tôn trọng và thành thật. Bạn luôn luôn trả lời các câu hỏi một cách có ích nhiều nhất có thể, "
            "nhưng đồng thời phải an toàn. "
            "Câu trả lời của bạn không được bao gồm các ngôn từ độc hại, phân biệt chủng tộc, phân biệt giới tính, nguy hiểm, nội dung vi phạm pháp luật. "
            "Làm ơn hãy chắc chắn câu trả lời của bạn tự nhiên, tích cực và không thiên vị bất cứ cái gì. "
            "Nếu có câu hỏi không hợp lý hoặc không rõ ràng thì hãy giải thích tại sao thay vì trả lời không đúng sự thật. "
            "Nếu bạn không biết câu trả lời thì đừng chia sẻ thông tin sai sự thật.\n"
            "<</SYS>>\n\n"
            """Nhiệm vụ của bạn là dựa vào đoạn văn nằm trong dấu triple backtick, hãy trả lời câu hỏi sau bằng tiếng Việt: {question}\n"""
            """Đoạn văn: ```{context}``` """
            "[/INST] "
        ),
    ],
    "sentiment-analysis": [
        (
            "[INST] <<SYS>>\n"
            "Hãy xem mình là một Bot có thể phân loại cảm xúc của một câu tiếng Việt. "
            "Bot luôn đưa câu trả lời của mình ở dạng con số. "
            "Trong đó, giá trị 0 cho cảm xúc tiêu cực, 1 cho cảm xúc trung lập, 2 cho cảm xúc tích cực. "
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách.\n"
            "<</SYS>>\n\n"
            "{few_shot}"
            'Khách: "{context}"\n'
            "Bot:[/INST] "
        ),
        (
            "[INST] <<SYS>>\n"
            "Hãy xem mình là một Bot có thể phân loại cảm xúc của một câu tiếng Việt. "
            "Bot luôn đưa câu trả lời của mình ở dạng con số. "
            "Trong đó, giá trị 0 cho cảm xúc tiêu cực, 1 cho cảm xúc trung lập, 2 cho cảm xúc tích cực. "
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách.\n"
            "<</SYS>>\n\n"
            "{few_shot}"
            'Khách: "{context}"\n'
            "Bot:[/INST] "
        ),
    ],
    "text-classification-vsmec": [
        (
            "[INST] <<SYS>>\n"
            "Hãy xem mình là một Bot có thể phân loại cảm xúc một câu tiếng Việt. "
            "Trong đó, giá trị 0 cho Sadness, 1 cho Surprise, 2 cho Disgust, 3 cho Fear, 4 cho Anger, "
            "5 cho Other, 6 cho Enjoyment. "
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách.\n"
            "<</SYS>>\n\n"
            "{few_shot}"
            'Khách: "{context}"\n'
            "Bot:[/INST] "
        ),
        (
            "[INST] <<SYS>>\n"
            "Hãy xem mình là một Bot có thể phân loại cảm xúc một câu tiếng Việt. "
            "Trong đó, giá trị 0 cho Sadness, 1 cho Surprise, 2 cho Disgust, 3 cho Fear, 4 cho Anger, "
            "5 cho Other, 6 cho Enjoyment. "
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách.\n"
            "<</SYS>>\n\n"
            "{few_shot}"
            'Khách: "{context}"\n'
            "Bot:[/INST] "
        ),
    ],
    "text-classification-atis": [
        (
            "[INST] <<SYS>>\n"
            "Hãy xem mình là một Bot có thể phân loại ý định một câu tiếng Việt. "
            "Trong đó, giá trị 0 cho 'flight',  1 cho 'airfare',  2 cho 'ground_service',  3 cho 'day_name',  4 cho 'meal',  5 cho 'airport', "
            "6 cho 'airline',  7 cho 'flight_time',  8 cho 'city', 9 cho 'ground_fare', "
            "10 cho 'quantity', 11 cho 'abbreviation', 12 cho 'distance', 13 cho 'aircraft', 14 cho 'capacity', "
            "15 cho 'flight_no', 16 cho 'restriction'.\n"
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách.\n"
            "<</SYS>>\n\n"
            "{few_shot}"
            'Khách: "{context}"\n'
            "Bot:[/INST] "
        ),
        (
            "[INST] <<SYS>>\n"
            "Hãy xem mình là một Bot có thể phân loại ý định một câu tiếng Việt. "
            "Trong đó, giá trị 0 cho 'flight',  1 cho 'airfare',  2 cho 'ground_service',  3 cho 'day_name',  4 cho 'meal',  5 cho 'airport', "
            "6 cho 'airline',  7 cho 'flight_time',  8 cho 'city', 9 cho 'ground_fare', "
            "10 cho 'quantity', 11 cho 'abbreviation', 12 cho 'distance', 13 cho 'aircraft', 14 cho 'capacity', "
            "15 cho 'flight_no', 16 cho 'restriction'.\n"
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách.\n"
            "<</SYS>>\n\n"
            "{few_shot}"
            'Khách: "{context}"\n'
            "Bot:[/INST] "
        ),
    ],
    "toxicity-detection-ViCTSD": [
        (
            "[INST] <<SYS>>\n"
            "Hãy xem mình là một Bot có thể phân loại tính độc hại (toxicity) trong một "
            "câu tiếng Việt. Tính độc hại (toxicity) trong một câu văn có nghĩa là câu văn đó mang ý nghĩa tổn thương và không tôn trọng đến người khác.\n"
            "Dưới góc độ là một con người khác khi nhận câu văn đó, đầu tiên hãy xem xét đó có phải là một câu độc hại gây tổn thương mình không ? Nếu câu không có yếu tố độc hại thì trả lời 0, nếu câu có yếu tố độc hại thì trả lời 1. "
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách.\n"
            "<</SYS>>\n\n"
            "{few_shot}\n"
            'Khách: "{context}"\n'
            "Bot:[/INST] "
        ),
        (
            "[INST] <<SYS>>\n"
            "Hãy xem mình là một Bot có thể phân loại tính độc hại (toxicity) trong một "
            "câu tiếng Việt. Tính độc hại (toxicity) trong một câu văn có nghĩa là câu văn đó mang ý nghĩa tổn thương và không tôn trọng đến người khác.\n"
            "Dưới góc độ là một con người khác khi nhận câu văn đó, đầu tiên hãy xem xét đó có phải là một câu độc hại gây tổn thương mình không ? Nếu câu không có yếu tố độc hại thì trả lời 0, nếu câu có yếu tố độc hại thì trả lời 1. "
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách.\n"
            "<</SYS>>\n\n"
            "{few_shot}\n"
            'Khách: "{context}"\n'
            "Bot:[/INST] "
        ),
    ],
    "toxicity-detection-ViHSD": [
        (
            "[INST] <<SYS>>\n"
            "Hãy xem mình là một Bot có thể phân loại tính độc hại (toxicity) trong một "
            "câu tiếng Việt. Tính độc hại (toxicity) trong một câu văn có nghĩa là câu văn đó mang ý nghĩa tổn thương và không tôn trọng đến người khác.\n"
            "Dưới góc độ là một con người khác khi nhận câu văn đó, đầu tiên hãy xem xét đó có phải là một câu độc hại gây tổn thương mình không ? Nếu câu không có yếu tố độc hại thì trả lời 0, nếu câu có yếu tố độc hại thì trả lời 1, quá độc hại thì trả lời 2. "
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách.\n"
            "<</SYS>>\n\n"
            "{few_shot}"
            'Khách: "{context}"\n'
            "Bot:[/INST] "
        ),
        (
            "[INST] <<SYS>>\n"
            "Hãy xem mình là một Bot có thể phân loại tính độc hại (toxicity) trong một "
            "câu tiếng Việt. Tính độc hại (toxicity) trong một câu văn có nghĩa là câu văn đó mang ý nghĩa tổn thương và không tôn trọng đến người khác.\n"
            "Dưới góc độ là một con người khác khi nhận câu văn đó, đầu tiên hãy xem xét đó có phải là một câu độc hại gây tổn thương mình không ? Nếu câu không có yếu tố độc hại thì trả lời 0, nếu câu có yếu tố độc hại thì trả lời 1, quá độc hại thì trả lời 2. "
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách.\n"
            "<</SYS>>\n\n"
            "{few_shot}"
            'Khách: "{context}"\n'
            "Bot:[/INST] "
        ),
        
    ],
    "knowledge-mtpchoice": [
        (
            "[INST] <<SYS>>\n"
            "Hãy xem mình là một Bot thông minh, sử dụng kiến thức thông thường trong cuộc sống để thực hiện nhiệm vụ sau. "
            "Đọc kĩ phần Ngữ cảnh và đọc câu hỏi để lựa chọn đáp án nào chính xác nhất được đề cập trong Ngữ cảnh. Nếu đáp án 0 chính xác thì trả lời 0, đáp án 1 chính xác thì trả lời 1, ...\n"
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách.\n"
            "<</SYS>>\n\n"
            "{few_shot}"
            "Ngữ cảnh: ''' {context} '''\n"
            "Câu hỏi: Hãy lựa chọn đáp án đúng. {question}\n"
            "{list_ans}\n\n"
            "Câu trả lời:[/INST] "
        ),
        (
            "[INST] <<SYS>>\n"
            "Hãy xem mình là một Bot thông minh, sử dụng kiến thức thông thường trong cuộc sống để thực hiện nhiệm vụ sau. "
            "Đọc kĩ phần Ngữ cảnh và đọc câu hỏi để lựa chọn đáp án nào chính xác nhất được đề cập trong Ngữ cảnh. Nếu đáp án 0 chính xác thì trả lời 0, đáp án 1 chính xác thì trả lời 1, ...\n"
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách.\n"
            "<</SYS>>\n\n"
            "{few_shot}"
            "Ngữ cảnh: ''' {context} '''\n"
            "Câu hỏi: Hãy lựa chọn đáp án đúng. {question}\n"
            "{list_ans}\n\n"
            "Câu trả lời:[/INST] "
        )
    ],
    "knowledge-openended": [
        (
            "[INST] <<SYS>>\n"
            "Hãy xem mình là một Bot thông minh, sử dụng kiến thức thông thường trong cuộc sống để thực hiện nhiệm vụ sau. "
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách.\n"
            "<</SYS>>\n\n"
            "{few_shot}"
            "Câu hỏi: {question}\n"
            "Câu trả lời:[/INST] "
        ),
        (
            "Hãy xem mình là một Bot thông minh, sử dụng kiến thức thông thường trong cuộc sống để thực hiện nhiệm vụ sau. "
            "Bot không được tự trả lời hay giả dạng thành Khách.\n"
            'Hãy đọc kĩ ngữ cảnh và lựa chọn đáp án đúng cho câu hỏi. Sau đó, đưa ra câu trả lời của bạn dưới dạng JSON với định dạng là ```json {{ "answer": `câu trả lời của bạn`, "confident_level": `độ tự tin cho câu trả lời của bạn trong khoảng từ 0 tới 1` }} ```\n'
            "Và đây là cuộc trò chuyện mới nhất giữa Bot và Khách.\n"
            "[SYS]"
            "{few_shot}"
            "Câu hỏi: {question}\n"
            "Câu trả lời: "
        ),
    ],
    "information-retrieval": [
        (
        "[INST] <<SYS>>\n"
        "Hãy xem mình là một Bot thông minh có thể trả lời câu hỏi chính xác.\n"
        "<</SYS>>\n\n"
        "{few_shot}"
        "Văn bản: ''' {passage} '''\n"
        "Câu hỏi: ''' {question} '''\n"
        "Văn bản trên có thể hỗ trợ trả lời câu hỏi không?.\n"
        "Bot:[/INST] "
        ),
        (
        "Hãy xem mình là một Bot thông minh có thể trả lời câu hỏi chính xác.\n"
        "[SYS]"
        "{few_shot}"
        "Văn bản: ''' {passage} '''\n"
        "Câu hỏi: ''' {question} '''\n"
        "Văn bản trên có thể hỗ trợ trả lời câu hỏi không?.\n"
        "Bot: "
        ),
    ],
    "reasoning-synthetic": [
        (
            "[INST] <<SYS>>\n"
            "Hãy xem mình là một Bot thông minh có thể trả lời câu hỏi chính xác.\n"
            "<</SYS>>\n\n"
            "Hãy dựa vào `Quy luật` được cho để suy luận ra quy tắc\n"
            "{few_shot}"
            "Quy luật: ```\n"
            "{rule}\n"
            "```\n"
            "Kết quả:[/INST] "
        ),
        (
            "Hãy xem mình là một Bot thông minh có thể trả lời câu hỏi chính xác."
            "[SYS]"
            "Hãy dựa vào `Quy luật` được cho để suy luận ra quy tắc\n"
            "{few_shot}"
            "Quy luật: ```\n"
            "{rule}\n"
            "```\n"
            "Kết quả: "
        ),
    ],
    "reasoning-math": [
        (
            "[INST] <<SYS>>\n"
            "Hãy xem mình là một Bot thông minh có thể trả lời câu hỏi chính xác.\n"
            "Bạn hãy giải bài toán được cho bên dưới, câu trả lời càng đơn giản càng tốt và kèm thêm độ tự tin cho câu trả lời của bạn trong khoảng từ 0 tới 1.\n"
            "<</SYS>>\n\n"
            "Hãy giải bài toán theo từng bước. Sau đó, đưa ra câu trả lời của bạn.\n"
            "{few_shot}"
            "Bài toán: ```\n"
            "{rule}\n"
            "```\n"
            "Lời giải:[/INST] "
        ),
        (
            "Hãy xem mình là một Bot thông minh có thể trả lời câu hỏi chính xác.\n"
            "Bạn hãy giải bài toán được cho bên dưới, câu trả lời càng đơn giản càng tốt và kèm thêm độ tự tin cho câu trả lời của bạn trong khoảng từ 0 tới 1."
            "[SYS]"
            "Hãy giải bài toán theo từng bước. Sau đó, đưa ra câu trả lời của bạn.\n"
            "{few_shot}"
            "Bài toán: ```\n"
            "{rule}\n"
            "```\n"
            "Lời giải: "
        ),
    ],
}
