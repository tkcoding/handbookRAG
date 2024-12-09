class promptTemplate(object):
    def __init__(self):
        pass

    def LOPromptWithContext(self) -> str:
        return """
            Here is example of a table of contents in JSON format for some course about python data science for absolute beginner for 10 weeks allocated time.
            ###
            {{'Learning Outcomes': {{
            'I. Data Foundations':
            {{
            'Duration':2 weeks,
            'A. Define the workflow, tools and approaches data scientists use to analyse data': {{}},
            'B. Apply the Data Science Workflow to solve a task': {{}},
            'C. Navigate through directories using the command line': {{}},
            'D. Conduct arithmetic and string operations in Python': {{}}}},
            'II. Working with Data':
            {{
            'Duration': 2 weeks,
            'A. Use DataFrames and Series to read data: {{}},
            'B. Define key principles of data visualization': {{}},
            'C. Create line plots, bar plots , histograms and box plots using Seaborn and Matplotlib': {{}},
            'D. Determine causality and sampling bias': {{}}}},
            'III. Data Science Modeling':
            'Duration': 3 weeks
            {{'A. Define data modeling and linear regression': {{}},
            'B. Describe errors of bias and variance': {{}},
            'C. Build a k-nearest neighbors model using the scikit-learn library': {{}},
            'D. Evaluate a model using metrics such as classification accuracy/error, confusion matrix, ROC/AOC curves and loss functions': {{}}}},
            'IV. Data Science Applications':
            'Duration' : 3 weeks
            {{'A. Demonstrate how to tokenize natural language text': {{}},
            'B. Perform text classification model using scikit-learn, CountVectorizer, TfidfVectorizer, and TextBlog': {{}},
            'C. Create rolling means and plot time series data': {{}},
            'D. Explore an additional data science topic based on class interest. Options include: clustering, decision trees, robust regression and deploying model with Flask': {{}}}},
            'E. Final Project: Complete a capstone project on data science real world application': {{}}}}}}
            ###

            Your task it to estimate each section with total duration needed , the amount of all section should add up to allocated time according to the template provided above.
            Learning outcome from document provided MUST BE used as the main driving point of the constructed learning outcoming with addition provided information.
            1. Scale the duration needed for each section with respect to the complexity and expertise level.

            2. Select appropriate Bloom’s Taxonomy levels depending of the allocated time provided. DO NOT include additional level if the allocated time is shorter than the guidelines below.
            - If the course is less than 1 week, focus on **Remember** and **Understand** levels only.
            - If the course is between 1–3 weeks, include **Apply** and **Analyze** levels.
            - If the course is more 3 weeks, include **Evaluate** and **Create** levels.

            Learning outcome from document:
            {context}

            Generate learning objectives in the JSON format for a course about {course_topic} for {target_audience}.
            course description is as follows: {course_description}.
            Pre-requesite : {pre_requisites}
            Allocated time : {allocated_time}
            Response should be presented in {language}

            No explanation or additional stating on bloom's taxonomy level needed in response.
        """

    def LOPromptWithoutContext(self) -> str:
        return """
            Here is example of a table of contents in JSON format for some course about python data science for absolute beginner for 10 weeks allocated time.
            ###
            {{'Learning Outcomes': {{
            'I. Data Foundations':
            {{
            'Duration':2 weeks,
            'A. Define the workflow, tools and approaches data scientists use to analyse data': {{}},
            'B. Apply the Data Science Workflow to solve a task': {{}},
            'C. Navigate through directories using the command line': {{}},
            'D. Conduct arithmetic and string operations in Python': {{}}}},
            'II. Working with Data':
            {{
            'Duration': 2 weeks,
            'A. Use DataFrames and Series to read data: {{}},
            'B. Define key principles of data visualization': {{}},
            'C. Create line plots, bar plots , histograms and box plots using Seaborn and Matplotlib': {{}},
            'D. Determine causality and sampling bias': {{}}}},
            'III. Data Science Modeling':
            'Duration': 3 weeks
            {{'A. Define data modeling and linear regression': {{}},
            'B. Describe errors of bias and variance': {{}},
            'C. Build a k-nearest neighbors model using the scikit-learn library': {{}},
            'D. Evaluate a model using metrics such as classification accuracy/error, confusion matrix, ROC/AOC curves and loss functions': {{}}}},
            'IV. Data Science Applications':
            'Duration' : 3 weeks
            {{'A. Demonstrate how to tokenize natural language text': {{}},
            'B. Perform text classification model using scikit-learn, CountVectorizer, TfidfVectorizer, and TextBlog': {{}},
            'C. Create rolling means and plot time series data': {{}},
            'D. Explore an additional data science topic based on class interest. Options include: clustering, decision trees, robust regression and deploying model with Flask': {{}}}},
            'E. Final Project: Complete a capstone project on data science real world application': {{}}}}}}
            ###

            Your task it to estimate each section with total duration needed , the amount of all section should add up to allocated time according to the template provided above.
            1. Scale the duration needed for each section with respect to the complexity and expertise level.

            2. Select appropriate Bloom’s Taxonomy levels depending of the allocated time provided. DO NOT include additional level if the allocated time is shorter than the guidelines below.
            - If the course is less than 1 week, focus on **Remember** and **Understand** levels only.
            - If the course is between 1–3 weeks, include **Apply** and **Analyze** levels.
            - If the course is more 3 weeks, include **Evaluate** and **Create** levels.

            Generate learning objectives in the JSON format for a course about {course_topic} for {target_audience}.
            course description is as follows: {course_description}.
            Pre-requesite : {pre_requisites}
            Allocated time : {allocated_time}

            Response should be presented in {language}
            No explanation needed in response.
        """
