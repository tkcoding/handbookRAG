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
