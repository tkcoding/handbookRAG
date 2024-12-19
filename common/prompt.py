class promptTemplate(object):
    def __init__(self):
        pass

    def LOPromptWithContext(self) -> str:
        return """
            Your task it to estimate each section with total duration needed , the amount of all section should add up to allocated time according to the template provided above.
            Learning outcome from document provided MUST BE used as the main driving point of the constructed learning outcome with addition provided information.
            Learning outcome from document:
            {context}

            Generate learning objectives in the JSON format for a course about {course_topic} for {target_audience}.
            course description is as follows: {course_description}.
            Pre-requesite : {pre_requisites}
            Allocated time : {allocated_time}
            Response should be presented in {language}

            1. Scale the duration needed for each section with respect to the complexity and expertise level.

            2. According to the allocated time. Include bloom taxonomy level respectively.
            - If the allocated time is less than 1 week, focus on **Remember** and **Understand** levels ONLY.
            - If the allocated time is between 1–3 weeks, include **Remember** , **Understand** , **Apply** and **Analyze** ONLY.
            - If the allocated time is more 3 weeks, include **Remember** , **Understand** , **Apply** , **Analyze**, **Evaluate** and **Create** levels ONLY.


            3. Generate 3–7 concise, measurable LOs aligned with the course content.

            4. Format the output as:
            - LO1: [Action Verb] + [Specific Knowledge/Skill]
            - LO2: ...
            - LO3: ...
            Reponse only includes learning outcomes (LO).
            No explanation or additional stating on bloom's taxonomy level needed in response.
        """

    def LOPromptWithoutContext(self) -> str:
        return """
            Your task it to estimate each section with total duration needed , the amount of all section should add up to allocated time according to the template provided above.
            Learning outcome from document provided MUST BE used as the main driving point of the constructed learning outcome with addition provided information.

            Generate learning objectives in the JSON format for a course about {course_topic} for {target_audience}.
            course description is as follows: {course_description}.
            Pre-requesite : {pre_requisites}
            Allocated time : {allocated_time}
            Response should be presented in {language}

            1. Scale the duration needed for each section with respect to the complexity and expertise level.

            2. According to the allocated time. Include bloom taxonomy level respectively.
            - If the allocated time is less than 1 week, focus on **Remember** and **Understand** levels ONLY.
            - If the allocated time is between 1–3 weeks, include **Remember** , **Understand** , **Apply** and **Analyze** ONLY.
            - If the allocated time is more 3 weeks, include **Remember** , **Understand** , **Apply** , **Analyze**, **Evaluate** and **Create** levels ONLY.


            3. Generate 3 to 7 concise, measurable LOs aligned with the course content.

            4. Format the output as:
            - LO1: [Action Verb] + [Specific Knowledge/Skill]
            - LO2: ...
            - LO3: ...
            Reponse only includes learning outcomes (LO).
            No explanation or additional stating on bloom's taxonomy level needed in response.
        """

    def LORephraseBetaPrompt(self) -> str:
        return """
         You are a curriculum expert in creating learning outcomes according to bloom taxonomy and context provided.

        1. Given allocated time of {allocated_time} first decide what level should be included.
        According to the allocated time. Include bloom taxonomy level respectively.
            - If the allocated time is less than 1 week, focus on **Remember** and **Understand** levels ONLY.
            - If the allocated time is between 1–3 weeks, include **Remember** , **Understand** , **Apply** and **Analyze** ONLY.
            - If the allocated time is more 3 weeks, include **Remember** , **Understand** , **Apply** , **Analyze**, **Evaluate** and **Create** levels ONLY.

        2. Revise all learning outcomes and rephrase according to the bloom taxanomy guidelines provided above.
        3. Allocated time can be in hours , weeks , months or mixture of any of it.
        4. Generate 3–10 concise, measurable LOs aligned with the course content depending on the levels provided.

        Provide learning outcomes categorise into each level:
        for e.g. for less than 1 week , it should include remember and understand levels.
        Remember:
        - Able to recall data structure and data types.

        Understand:
        - Comprehend basic data types usage

        Generate learning objectives in the JSON format for a course about {course_topic} for {target_audience}.
        course description is as follows: {course_description}.
        Pre-requesite : {pre_requisites}
        Response should be presented in {language}

        """

    def CourseStructurePrompt(self) -> str:
        return """
            You are a curriculum expert who design course structure for different learner.
            Your task is to create course structure according to the information provided.

            Course structure should use the learning outcomes as the main driving point.

            Learning outcome:
            {learning_outcome}

            Generate course structure in the JSON format for a course about {course_topic} for {target_audience}.
            course description is as follows: {course_description}.
            Pre-requesite : {pre_requisites}
            Allocated time : {allocated_time}
            Response should be presented in {language}

            1.All learning outcome provided must be included as a consideration of creating course structure.

            2. Course structure presented should be able to complete within allocated time with its complexity.
            DO NOT provide unreasonable course structure for example 10 course structures for an allocated time of 2 days which is totally unreasonable.

            Reponse only includes course struture.
            No explanation needed in response.
        """

    def BloomReviewerPrompt(self) -> str:
        return """
        You are a reviewer and your task is to revise and rephrase learning outcomes provided with guidelines below:
        1. Given allocated time of {allocated_time} first decide what level should be included.
        According to the allocated time. Include bloom taxonomy level respectively.
            - If the allocated time is less than 1 week, focus on **Remember** and **Understand** levels ONLY.
            - If the allocated time is between 1–3 weeks, include **Remember** , **Understand** , **Apply** and **Analyze** ONLY.
            - If the allocated time is more 3 weeks, include **Remember** , **Understand** , **Apply** , **Analyze**, **Evaluate** and **Create** levels ONLY.

        2. Revise all learning outcomes and rephrase according to the bloom taxanomy guidelines provided above.
        3. Allocated time can be in hours , weeks , months or mixture of any of it.

        Review and revise the following learning outcomes:
        {learning_outcomes}

        Provide learning outcomes categorise into each level:
        for e.g. for less than 1 week , it should include remember and understand levels.
        Remember:
        - Able to recall data structure and data types.

        Understand:
        - Comprehend basic data types usage.

        """

    def llamaparsePrompt(self) -> str:
        return """
        You are a highly proficient language model designed to convert pages from PDF, PPT and other files into structured markdown text. Your goal is to accurately transcribe text, represent formulas in LaTeX MathJax notation, and identify and describe images, particularly graphs and other graphical elements.

        You have been tasked with creating a markdown copy of each page from the provided PDF or PPT image. Each image description must include a full description of the content, a summary of the graphical object.

        Maintain the sequence of all the elements.

        For the following element, follow the requirement of extraction:
        for Text:
        - Extract all readable text from the page.
        - Exclude any diagonal text, headers, and footers.
        - DO NOT summary the original text.

        for Text which includes hyperlink:
            -Extract hyperlink and present it with the text

        for Formulas:
        - Identify and convert all formulas into LaTeX MathJax notation.

        for Image Identification and Description:
        - Identify all images, graphs, and other graphical elements on the page.
        - If the image has graph , extract the graph as image . DO NOT convert it into a table or extract the wording inside the graph.
        - If the image has a subtitle or caption, include it in the description.
        - If the image has a formula convert it into LaTeX MathJax notation.
        - If the image has a organisation chart , convert it into a hierachical understandable format.
        - If the image contain process flow , capture it as a whole image instead of separate into blocks of images.

        for Table:
        - Try to retain the columns and structure of the table and extract it into markdown format.

        # OUTPUT INSTRUCTIONS

        - Ensure all formulas are in LaTeX MathJax notation.
        - Exclude any diagonal text, headers, and footers from the output.
        - For each image and graph, provide a detailed description,caption if there's any and summary.
        """
