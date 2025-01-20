class promptTemplate(object):
    def __init__(self):
        pass

    def LOPromptWithContext(self) -> str:
        return """
            Your task it to estimate each section with total duration needed , the amount of all section should add up to allocated time according to the template provided above.

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

            Learning outcome from document provided MUST BE used as the main driving point of the constructed learning outcome with addition provided information.
            Learning outcome from document:
            {context}

            Generate learning objectives in the JSON format for a course about {course_topic} for {target_audience}.
            course description is as follows: {course_description}.
            Pre-requesite : {pre_requisites}
            Allocated time : {allocated_time}
            Response should be presented in {language}
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

    def LOCourseStructureOnePrompt(self) -> str:
        return """
        You are a curriculum expert in developing educational courses using Bloom’s taxonomy and context provided.
        Generate learning objectives in the bullet format for a course about {course_topic} for {target_audience} for allocated time of {allocated_time}.
        course description is as follows: {course_description}.
        Pre-requesite : {pre_requisites}
        Allocated time : {allocated_time}
        Response should be presented in {language}
        Learning outcome from document provided MUST BE used as the main driving point of the constructed learning outcome with addition provided information.
        Learning outcome from document: {context}

        If allocated time is presented in credits, weeks, months, semesters or years, use these rules to convert it to hours:
        1 credit = 10 hours of lectures or seminars and 20 hours of preparation time
        1 credit hour = 1 hour of class work and 2 hours of preparation time
        1 week = 3 hours of class time and 6 hours of preparation time
        1 semester = 45 hours of class time and 90 hours of student preparation for a 3 credit course
        Generate 3–7 concise, measurable LOs aligned with the course content, target audience and Bloom’s taxonomy.

        Format the output as:

        LO1:
            Learning Outcome: [Action Verb using Bloom’s Taxonomy] + [Specific Knowledge/Skill]
            Reasoning : ...
        LO2:
            Learning Outcome : Remember the fundamental concepts of matrix operations, including addition, subtraction, and multiplication.
            Reasoning : This outcome is essential as it establishes a foundational understanding of matrix operations, which are critical for further applications in electrical engineering.
        LO3: ...
        For each learning outcome (LO) add reasoning, why this learning outcome is important in this course."""

    def LORephraseBetaPrompt(self) -> str:
        return """
         You are a curriculum expert in creating learning outcomes according to bloom taxonomy and context provided.

        1. Given allocated time, first decide what level should be included.

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
        allocated_time : {allocated_time}
        course description : {course_description}.
        Pre-requesite : {pre_requisites}
        Response should be presented in {language}

        """

    def CourseStructurePromptRevised(self) -> str:
        return """
        Create several modules for the course according to the learning outcomes generated:
        {learning_outcome}

        It's crucial that the module created should be within {allocated_time}.

        Align every module of the course with corresponding LO.
        When generating course structure and describing its’ agenda mention concrete content that should be taught in this module. This content should reflect the most important information to master the LO.
        Estimate each module of the course with duration of hours needed, the amount of all modules should add up to allocated time according to the template provided above. Scale the duration needed for each module with respect to the complexity and expertise level. Use minutes or hours to describe duration of each module of the course.
        For lower level LO’s Remember , Understand , Apply) sections should be shorter, than for higher-level LO’s (Analyze, Evaluate and Create).
        Separately indicate how much time of each module will be devoted to work in class, and how much to independent work of the learner at home (preparation time).

        For each module of the course suggest teaching methods, practical tasks and potential assessments that a teacher should use to achieve the goal. To select methods, practical tasks and potential assessments use the best practices according to educational science.
        """

    def CourseStructurePrompt(self) -> str:
        return """
            You are a curriculum expert who design course structure for different learner.
            Your task is to create course structure according to the information provided.

            Course structure should use the learning outcomes as the main driving point.

            Learning outcome:
            {learning_outcome}

            Generate course structure in the bullet format for a course about {course_topic} for {target_audience}.
            course description is as follows: {course_description}.
            Pre-requesite : {pre_requisites}
            Allocated time : {allocated_time}
            Response should be presented in {language}


            Below are the requirements for course structure generation:
            1. The Structure should contain 3 elements:
                - Lesson (micro-learning activity between one and three hours of work)
                - Topic (collection of few lessons, typically for several hours of work)
                - Module (learning activity around one content e.g. historical period, might consist of one or more topics)
            2. The course element should contain:
                - name for course element
                - agenda (brief text overview, description of the suggested content for the element)
            3. Multiple learning outcomes can be mapped to single lesson.
            4. All learning outcome should be mapped to at least one of the lesson created.

            Glossary:
                - Module - learning activity around one content e.g. historical period, might consist of one or more topics. On this level teachers and students are working on mastery of one piece of content through different taxonomy levels.
                        During this learning activity students can achieve middle-range educational goals associated with first four or five levels of Bloom taxonomy (remember, understand, apply, analyze, evaluate).
                        Typically learning outcome on this level target apply, analyze, evaluate levels.
                - Topic - a collection of few lessons, typically for several hours of work. Typically learning outcome on this level target remember, understand, apply and analyze levels of Bloom's taxonomy.
                - Lesson - micro-learning activity between one and three hours of work. Typically learning outcome on this level target remember, understand and apply levels of Bloom's taxonomy.

            Output structure should be as follows:

                Module : Matrix Operations in Electrical Engineering
                - Agenda :
                    - Topics : Introduction to Matrices
                    - Agenda :
                        - Lessons : Key Matrix Terms
                        - Agenda :
                        - Learning outcomes associated : LO1
                    - Topics : Matrix Operations
                    - Agenda :
                        - Lessons : Significance of Matrix Operations
                        - Agenda :
                        - Learning outcomes associated : LO2
                                                       : LO3

            Explanation on output structure : Module should be the parents for topics and lessons should be the parents for lessons.
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
