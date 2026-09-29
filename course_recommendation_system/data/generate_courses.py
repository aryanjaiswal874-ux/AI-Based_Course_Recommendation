"""
data/generate_courses.py
------------------------
Builds data/courses.csv, a CLEARLY LABELLED SAMPLE dataset of 112 illustrative
online courses (8 per domain, 14 domains).

IMPORTANT
- Course titles/descriptions are generic and illustrative. They are NOT official
  listings and no course is claimed to be affiliated with or certified by any platform.
- Ratings and enrollments are synthetic (seeded random numbers) for demonstration.
- Course URLs are platform search/landing links, not pinned to a specific course.

Run:  python data/generate_courses.py
"""
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd

# (name, description, skills ';'-separated, difficulty, hours, prerequisites)
# Each list is ordered from foundational to advanced -> used for roadmaps.
CATALOG = {
    ("Data Structures & Algorithms", "Software Engineer"): [
        ("Data Structures Foundations: Arrays and Strings", "Learn arrays, strings, hashing and complexity analysis through interview-style problems.", "Arrays;Strings;Hashing;Big-O Analysis;Problem Solving", "Beginner", 18, "Basic programming in any language"),
        ("Linked Lists, Stacks and Queues", "Implement and apply linear data structures with pointer-manipulation drills.", "Linked Lists;Stacks;Queues;Data Structures;Problem Solving", "Beginner", 16, "Arrays and strings basics"),
        ("Sorting, Searching and Two-Pointer Techniques", "Sorting algorithms, binary search variants, sliding window and two-pointer patterns.", "Sorting;Binary Search;Two Pointers;Sliding Window;Algorithms", "Intermediate", 15, "Arrays and strings"),
        ("Recursion and Backtracking Patterns", "Master recursion trees, memoisation basics and classic backtracking problems.", "Recursion;Backtracking;Algorithms;Problem Solving", "Intermediate", 14, "Linked lists and basic programming"),
        ("Trees and Binary Search Trees", "Traversals, BST operations, heaps and tries with worked examples.", "Trees;Binary Search Trees;Heaps;Tries;Data Structures", "Intermediate", 20, "Recursion basics"),
        ("Graph Algorithms: BFS, DFS and Shortest Paths", "Graph representations, traversal, Dijkstra, topological sort and union-find.", "Graphs;BFS;DFS;Dijkstra;Algorithms", "Intermediate", 22, "Trees and queues"),
        ("Dynamic Programming Masterclass", "Move from recursion to tabulation across knapsack, LCS and grid problems.", "Dynamic Programming;Recursion;Algorithms;Optimization", "Advanced", 24, "Recursion and graphs"),
        ("Advanced Algorithms and Interview Preparation", "Segment trees, advanced graph techniques, greedy proofs and timed mock interviews.", "Segment Trees;Advanced Graphs;Greedy Algorithms;Problem Solving;Algorithms", "Advanced", 28, "Dynamic programming"),
    ],
    ("Artificial Intelligence", "AI Engineer"): [
        ("Introduction to Artificial Intelligence", "Core AI concepts, intelligent agents, problem formulation and real-world applications.", "Artificial Intelligence;Intelligent Agents;Search Algorithms;Python", "Beginner", 20, "Basic programming"),
        ("Prompt Engineering and LLM Basics", "How large language models work and how to write effective prompts for practical tasks.", "Prompt Engineering;LLMs;Generative AI;Python", "Beginner", 10, "None"),
        ("Responsible AI and AI Ethics", "Bias, fairness, privacy, transparency and governance in AI systems.", "AI Ethics;Fairness;Privacy;Artificial Intelligence", "Beginner", 8, "None"),
        ("AI Problem Solving: Search and Knowledge Representation", "A* search, adversarial search, propositional logic and knowledge graphs for reasoning systems.", "Search Algorithms;A* Search;Knowledge Representation;Logic;Artificial Intelligence", "Intermediate", 22, "Introduction to AI; Python"),
        ("Computer Vision with OpenCV", "Image processing, feature detection, object detection and hands-on OpenCV projects.", "Computer Vision;OpenCV;Image Processing;Python;Deep Learning", "Intermediate", 24, "Python; NumPy"),
        ("Natural Language Processing with Transformers", "Tokenisation, embeddings, attention and fine-tuning transformer models for NLP tasks.", "NLP;Transformers;Embeddings;Deep Learning;Python", "Advanced", 30, "Deep learning basics"),
        ("Building LLM Applications with Retrieval-Augmented Generation", "Design RAG pipelines with vector search, chunking and evaluation.", "LLMs;RAG;Vector Databases;Prompt Engineering;Python", "Advanced", 20, "Python; REST APIs"),
        ("Reinforcement Learning Fundamentals", "Markov decision processes, Q-learning and policy gradients with simulated environments.", "Reinforcement Learning;Q-Learning;Policy Gradients;Python;Machine Learning", "Advanced", 28, "Machine learning; probability"),
    ],
    ("Machine Learning", "Machine Learning Engineer"): [
        ("Machine Learning Foundations: Regression and Classification", "Core supervised learning ideas with linear regression, logistic regression and evaluation metrics.", "Machine Learning;Regression;Classification;Python;Statistics", "Beginner", 25, "Python basics; school-level maths"),
        ("Supervised Learning with Scikit-learn", "Train, compare and validate decision trees, SVMs and ensembles with scikit-learn.", "Scikit-learn;Supervised Learning;Model Evaluation;Pandas;Machine Learning", "Intermediate", 22, "Python; Pandas"),
        ("Feature Engineering and Model Tuning", "Encoding, scaling, feature selection, cross-validation and hyperparameter search.", "Feature Engineering;Hyperparameter Tuning;Cross-Validation;Pandas;Machine Learning", "Intermediate", 18, "Supervised learning basics"),
        ("Unsupervised Learning: Clustering and Dimensionality Reduction", "K-means, hierarchical clustering, DBSCAN and PCA applied to real datasets.", "Clustering;PCA;Unsupervised Learning;Scikit-learn;Machine Learning", "Intermediate", 16, "Machine learning foundations"),
        ("Deep Learning with TensorFlow and Keras", "Build neural networks, CNNs and RNNs with the Keras API.", "Deep Learning;Neural Networks;TensorFlow;Keras;Machine Learning", "Intermediate", 30, "Machine learning foundations; Python"),
        ("Deep Learning with PyTorch", "Tensors, autograd, CNNs and training loops written in PyTorch.", "PyTorch;Deep Learning;CNN;Neural Networks", "Advanced", 30, "Deep learning basics"),
        ("MLOps: Deploying Machine Learning Models", "Package, serve and monitor ML models using Docker and REST APIs.", "MLOps;Docker;Model Deployment;REST APIs;Machine Learning", "Advanced", 20, "Python; basic Docker"),
        ("Advanced Machine Learning Projects and Capstone", "End-to-end projects with ensembles, XGBoost, time series and deep learning.", "Machine Learning;Ensemble Methods;XGBoost;Time Series;Deep Learning", "Advanced", 40, "Supervised learning; deep learning basics"),
    ],
    ("Data Science", "Data Scientist"): [
        ("Python for Data Science", "Python essentials for analysts: notebooks, data types, functions and basic analysis.", "Python;Data Analysis;Jupyter;Data Science", "Beginner", 15, "None"),
        ("NumPy and Pandas for Data Analysis", "Array computing, DataFrames, grouping, merging and data cleaning.", "NumPy;Pandas;Data Cleaning;Data Analysis", "Beginner", 18, "Python basics"),
        ("Statistics and Probability for Data Science", "Descriptive statistics, distributions, hypothesis testing and confidence intervals.", "Statistics;Probability;Hypothesis Testing;Data Science", "Beginner", 20, "School-level maths"),
        ("SQL for Data Analysis", "Query relational data with joins, aggregations and subqueries for analytics.", "SQL;Joins;Aggregations;Data Analysis", "Beginner", 14, "None"),
        ("Data Visualization with Matplotlib and Plotly", "Create static and interactive charts that communicate insights clearly.", "Matplotlib;Plotly;Data Visualization;Pandas", "Intermediate", 12, "Pandas basics"),
        ("Exploratory Data Analysis in Practice", "A repeatable workflow for profiling, cleaning and understanding new datasets.", "Exploratory Data Analysis;Pandas;Data Visualization;Statistics;Data Cleaning", "Intermediate", 14, "NumPy and Pandas"),
        ("Business Intelligence with Power BI and Excel", "Build dashboards and reports with Power BI and advanced Excel techniques.", "Power BI;Excel;Dashboards;Data Visualization", "Intermediate", 16, "Basic spreadsheets"),
        ("Big Data Analytics with Apache Spark", "Distributed data processing with PySpark DataFrames and Spark SQL.", "Apache Spark;PySpark;Big Data;SQL", "Advanced", 26, "Python; SQL"),
    ],
    ("Web Development", "Web Developer"): [
        ("HTML and CSS Essentials", "Structure and style web pages with semantic HTML, Flexbox and Grid.", "HTML;CSS;Responsive Design;Flexbox", "Beginner", 14, "None"),
        ("JavaScript Fundamentals", "Variables, functions, DOM manipulation, events and modern ES6 syntax.", "JavaScript;DOM;ES6;Problem Solving", "Beginner", 20, "HTML and CSS"),
        ("Responsive Web Design with Bootstrap and Tailwind CSS", "Build mobile-first layouts quickly using popular CSS frameworks.", "CSS;Bootstrap;Tailwind CSS;Responsive Design", "Beginner", 10, "HTML and CSS"),
        ("React: Building Modern User Interfaces", "Components, props, state, hooks and routing for single-page applications.", "React;JavaScript;Components;State Management;Hooks", "Intermediate", 24, "JavaScript"),
        ("Node.js and Express Backend Development", "Create REST APIs with Node.js, Express and MongoDB.", "Node.js;Express;REST APIs;MongoDB", "Intermediate", 22, "JavaScript"),
        ("Django Web Framework with Python", "Models, views, templates, authentication and deployment with Django.", "Django;Python;REST APIs;SQL", "Intermediate", 24, "Python basics"),
        ("Full-Stack Development with the MERN Stack", "Combine MongoDB, Express, React and Node.js into complete applications.", "MongoDB;Express;React;Node.js;Full-Stack", "Advanced", 40, "React; Node.js"),
        ("TypeScript and Web Performance Optimisation", "Type-safe front-end code, testing and Core Web Vitals optimisation.", "TypeScript;Web Performance;React;Testing", "Advanced", 18, "JavaScript; React"),
    ],
    ("App Development", "Mobile App Developer"): [
        ("Android App Development with Kotlin", "Activities, layouts, navigation and data storage in Android Studio.", "Android;Kotlin;Android Studio;Mobile UI", "Beginner", 28, "Basic programming"),
        ("Flutter and Dart: Cross-Platform Basics", "Build cross-platform apps with widgets, layouts and navigation.", "Flutter;Dart;Mobile UI;Widgets", "Beginner", 24, "Basic programming"),
        ("Building Modern UIs with Jetpack Compose", "Declarative Android UI, state handling and Material design.", "Jetpack Compose;Kotlin;Mobile UI;State Management", "Intermediate", 18, "Kotlin basics"),
        ("React Native for Mobile Developers", "Build iOS and Android apps with JavaScript and React Native.", "React Native;JavaScript;React;Mobile UI", "Intermediate", 26, "JavaScript; React basics"),
        ("iOS Development with Swift and SwiftUI", "Swift language essentials and SwiftUI interfaces for iPhone apps.", "Swift;SwiftUI;iOS;Mobile UI", "Intermediate", 28, "Basic programming"),
        ("Firebase Backend for Mobile Apps", "Authentication, Firestore and cloud functions for mobile back ends.", "Firebase;Authentication;Cloud Firestore;REST APIs", "Intermediate", 12, "Any mobile framework"),
        ("Advanced Flutter: State Management and Architecture", "Provider, Bloc, clean architecture and testing in Flutter.", "Flutter;Provider;Bloc;Clean Architecture;Testing", "Advanced", 22, "Flutter basics"),
        ("Publishing and Testing Mobile Apps", "Automated testing, CI/CD and store deployment for mobile releases.", "App Store Deployment;Mobile Testing;CI/CD;Performance", "Advanced", 14, "Any mobile framework"),
    ],
    ("Cyber Security", "Security Analyst"): [
        ("Cyber Security Fundamentals", "Threats, the CIA triad, risk management and basic defensive controls.", "Cyber Security;CIA Triad;Threat Modeling;Networking", "Beginner", 16, "None"),
        ("Computer Networks and Network Security", "TCP/IP, routing, firewalls, VPNs and intrusion detection basics.", "Networking;TCP/IP;Firewalls;Network Security", "Beginner", 22, "None"),
        ("Cryptography Essentials", "Symmetric and asymmetric encryption, hashing, digital signatures and PKI.", "Cryptography;Hashing;PKI;Encryption", "Intermediate", 18, "Basic maths"),
        ("Ethical Hacking and Penetration Testing", "Reconnaissance, scanning, exploitation and reporting in lab environments.", "Ethical Hacking;Penetration Testing;Kali Linux;Linux", "Intermediate", 30, "Networking; Linux"),
        ("Web Application Security and the OWASP Top 10", "Find and fix injection, XSS and authentication flaws in web apps.", "OWASP;SQL Injection;XSS;Web Security", "Intermediate", 20, "HTML; JavaScript basics"),
        ("Digital Forensics and Incident Response", "Evidence handling, disk and memory analysis and incident workflows.", "Digital Forensics;Incident Response;Log Analysis;Linux", "Intermediate", 24, "Operating systems basics"),
        ("Security Operations Centre (SOC) Analyst Skills", "Monitoring, SIEM alert triage and threat hunting fundamentals.", "SIEM;SOC;Threat Hunting;Log Analysis", "Advanced", 26, "Networking; security fundamentals"),
        ("Cloud Security and Secure Architecture", "Identity, encryption and zero-trust design for cloud workloads.", "Cloud Security;IAM;Zero Trust;AWS", "Advanced", 22, "Cloud basics; networking"),
    ],
    ("Cloud Computing", "Cloud Engineer"): [
        ("Cloud Computing Fundamentals", "Service models, virtualisation, deployment models and cloud economics.", "Cloud Computing;IaaS;PaaS;SaaS;Virtualization", "Beginner", 10, "None"),
        ("AWS Core Services Fundamentals", "Hands-on tour of EC2, S3, IAM, VPC and billing basics on AWS.", "AWS;EC2;S3;IAM", "Beginner", 16, "Cloud fundamentals"),
        ("Azure Cloud Fundamentals", "Virtual machines, storage, networking and identity on Microsoft Azure.", "Azure;Virtual Machines;Storage;Cloud Computing", "Beginner", 14, "Cloud fundamentals"),
        ("Google Cloud Platform Essentials", "Compute Engine, Cloud Storage and core GCP services for beginners.", "GCP;Compute Engine;Cloud Storage;Cloud Computing", "Beginner", 14, "Cloud fundamentals"),
        ("Serverless Applications with AWS Lambda", "Event-driven functions with Lambda, API Gateway and DynamoDB.", "AWS Lambda;Serverless;API Gateway;DynamoDB", "Intermediate", 16, "AWS fundamentals; Python or Node.js"),
        ("Cloud Networking and Security Basics", "VPC design, subnets, gateways, security groups and access control.", "VPC;Networking;IAM;Cloud Security", "Intermediate", 18, "Networking basics"),
        ("Cloud Architecture Design Patterns", "Design highly available, scalable and cost-aware cloud systems.", "Cloud Architecture;High Availability;Scalability;AWS", "Advanced", 24, "Any major cloud platform"),
        ("Data Engineering on the Cloud", "ETL pipelines, data warehouses and Spark jobs on cloud platforms.", "Data Engineering;ETL;BigQuery;Apache Spark", "Advanced", 28, "SQL; Python; cloud basics"),
    ],
    ("DevOps", "DevOps Engineer"): [
        ("Linux Command Line and Shell Scripting", "Navigate Linux, manage processes and automate tasks with Bash.", "Linux;Bash;Shell Scripting;Command Line", "Beginner", 14, "None"),
        ("Git and GitHub Workflows", "Version control, branching strategies, pull requests and collaboration.", "Git;GitHub;Version Control;Branching", "Beginner", 8, "None"),
        ("Docker and Containerisation", "Build images, run containers and orchestrate multi-container apps with Compose.", "Docker;Containers;Docker Compose;Linux", "Intermediate", 14, "Linux basics"),
        ("CI/CD Pipelines with GitHub Actions and Jenkins", "Automate build, test and deployment pipelines.", "CI/CD;GitHub Actions;Jenkins;Git", "Intermediate", 16, "Git; Docker basics"),
        ("Kubernetes for Beginners", "Pods, deployments, services and configuration on Kubernetes.", "Kubernetes;Container Orchestration;Docker;YAML", "Intermediate", 20, "Docker"),
        ("Infrastructure as Code with Terraform", "Provision cloud infrastructure declaratively with Terraform.", "Terraform;Infrastructure as Code;AWS;Cloud Computing", "Advanced", 18, "Cloud basics; Linux"),
        ("Monitoring and Observability with Prometheus and Grafana", "Metrics, dashboards, alerting and log aggregation for production systems.", "Prometheus;Grafana;Monitoring;Logging", "Advanced", 14, "Docker; Linux"),
        ("DevSecOps: Security in the Delivery Pipeline", "Integrate scanning, secrets management and policy checks into CI/CD.", "DevSecOps;Security Scanning;CI/CD;Docker", "Advanced", 16, "CI/CD; Docker"),
    ],
    ("Database Management", "Database Engineer"): [
        ("Database Management Systems Fundamentals", "Relational model, ER diagrams, keys and normalisation concepts.", "DBMS;ER Diagrams;Relational Model;Normalization", "Beginner", 16, "None"),
        ("SQL Basics: Queries and Joins", "SELECT, filtering, joins and aggregations with MySQL.", "SQL;Joins;Aggregations;MySQL", "Beginner", 14, "None"),
        ("Database Design and Normalisation", "Design schemas, apply normal forms and model real-world systems.", "Normalization;Database Design;ER Diagrams;SQL", "Intermediate", 12, "SQL basics"),
        ("Advanced SQL: Window Functions and Subqueries", "Window functions, CTEs, subqueries and analytical query patterns.", "SQL;Window Functions;Subqueries;CTEs", "Intermediate", 12, "SQL basics"),
        ("PostgreSQL for Developers", "Indexes, transactions, JSON support and administration with PostgreSQL.", "PostgreSQL;SQL;Indexes;Transactions", "Intermediate", 18, "SQL basics"),
        ("MongoDB and NoSQL Data Modelling", "Document modelling, aggregation pipelines and indexing in MongoDB.", "MongoDB;NoSQL;Aggregation Pipeline;Data Modeling", "Intermediate", 14, "Basic programming"),
        ("Query Optimisation and Indexing", "Read execution plans, tune queries and design effective indexes.", "Query Optimization;Indexes;Execution Plans;SQL", "Advanced", 14, "Advanced SQL"),
        ("Transactions, Concurrency and Database Administration", "ACID, isolation levels, locking, backup and recovery.", "Transactions;ACID;Concurrency Control;Database Administration", "Advanced", 18, "DBMS fundamentals; SQL"),
    ],
    ("Python Programming", "Python Developer"): [
        ("Python Programming Basics", "Variables, loops, functions and problem-solving with Python.", "Python;Variables;Loops;Functions", "Beginner", 16, "None"),
        ("Python Data Structures: Lists, Dictionaries and Sets", "Work confidently with Python's built-in collections.", "Python;Lists;Dictionaries;Data Structures", "Beginner", 10, "Python basics"),
        ("File Handling, Modules and Exceptions in Python", "Read and write files, organise code in modules and handle errors.", "Python;File Handling;Exceptions;Modules", "Beginner", 8, "Python basics"),
        ("Object-Oriented Programming in Python", "Classes, inheritance, polymorphism and design basics in Python.", "Python;OOP;Classes;Inheritance", "Intermediate", 14, "Python basics"),
        ("Python Intermediate: Generators, Decorators and Comprehensions", "Idiomatic Python features for cleaner, faster code.", "Python;Generators;Decorators;Comprehensions", "Intermediate", 12, "Python data structures"),
        ("Python Automation and Web Scraping", "Automate tasks, call REST APIs and scrape websites responsibly.", "Python;Web Scraping;Automation;REST APIs", "Intermediate", 14, "Python basics"),
        ("Testing and Debugging Python Applications", "Unit tests with pytest, mocking, debugging and code quality tools.", "Python;Unit Testing;pytest;Debugging", "Advanced", 12, "Python OOP"),
        ("Advanced Python: Concurrency and Performance", "Threading, multiprocessing, asyncio and profiling.", "Python;Multithreading;asyncio;Performance", "Advanced", 16, "Python intermediate"),
    ],
    ("Java Programming", "Java Developer"): [
        ("Java Programming Fundamentals", "Syntax, control flow, methods and arrays in Java.", "Java;Syntax;Control Flow;Methods", "Beginner", 20, "None"),
        ("Object-Oriented Programming with Java", "Classes, inheritance, interfaces and polymorphism in Java.", "Java;OOP;Inheritance;Polymorphism", "Beginner", 18, "Java basics"),
        ("Java Collections and Generics", "Lists, sets, maps, iterators and generic programming.", "Java;Collections Framework;Generics;Data Structures", "Intermediate", 14, "Java OOP"),
        ("Exception Handling and File I/O in Java", "Checked exceptions, streams and file handling.", "Java;Exceptions;File I/O;Streams", "Intermediate", 10, "Java OOP"),
        ("Multithreading and Concurrency in Java", "Threads, synchronisation, executors and concurrent utilities.", "Java;Multithreading;Concurrency;Synchronization", "Intermediate", 16, "Java OOP"),
        ("JDBC and Hibernate for Database Access", "Connect Java applications to relational databases with JDBC and Hibernate.", "Java;JDBC;Hibernate;SQL", "Intermediate", 16, "Java; SQL basics"),
        ("Spring Boot REST API Development", "Build production-style REST services with Spring Boot.", "Spring Boot;REST APIs;Java;Dependency Injection", "Advanced", 26, "Java OOP; SQL"),
        ("Microservices with Spring Cloud", "Service discovery, gateways, config and containerised microservices.", "Microservices;Spring Cloud;Docker;Java", "Advanced", 28, "Spring Boot"),
    ],
    ("C++ Programming", "C++ Developer"): [
        ("C++ Programming Basics", "Syntax, control flow, functions and arrays in C++.", "C++;Variables;Loops;Functions", "Beginner", 20, "None"),
        ("Pointers, References and Memory Management", "Understand pointers, dynamic allocation and common memory pitfalls.", "C++;Pointers;Memory Management;Dynamic Allocation", "Beginner", 14, "C++ basics"),
        ("Object-Oriented Programming in C++", "Classes, constructors, inheritance and polymorphism in C++.", "C++;OOP;Classes;Polymorphism", "Intermediate", 18, "C++ basics"),
        ("C++ Standard Template Library (STL)", "Vectors, maps, sets, iterators and STL algorithms.", "C++;STL;Vectors;Maps;Algorithms", "Intermediate", 14, "C++ basics"),
        ("Data Structures Implementation in C++", "Implement linked lists, trees and graphs from scratch in C++.", "C++;Data Structures;Linked Lists;Trees", "Intermediate", 22, "Pointers; C++ OOP"),
        ("Modern C++ (C++11 to C++20)", "Smart pointers, lambdas, move semantics and modern idioms.", "C++;Smart Pointers;Lambdas;Move Semantics", "Advanced", 18, "C++ OOP; STL"),
        ("Competitive Programming with C++", "Fast I/O, templates and contest-style algorithm practice.", "C++;Competitive Programming;Algorithms;Problem Solving", "Advanced", 26, "STL; algorithms"),
        ("Concurrency and Performance in C++", "Threads, atomics, profiling and optimisation techniques.", "C++;Multithreading;Performance;Concurrency", "Advanced", 20, "Modern C++"),
    ],
    ("UI/UX Design", "UI/UX Designer"): [
        ("UX Design Fundamentals", "Design thinking, personas, user flows and low-fidelity wireframes.", "UX Design;Design Thinking;User Personas;Wireframing", "Beginner", 14, "None"),
        ("Figma for Beginners", "Frames, components, auto layout and basic prototyping in Figma.", "Figma;UI Design;Prototyping;Auto Layout", "Beginner", 10, "None"),
        ("UI Design Principles: Colour, Typography and Layout", "Visual hierarchy, colour theory, typography and grid systems.", "UI Design;Typography;Color Theory;Layout", "Beginner", 12, "None"),
        ("User Research and Usability Testing", "Interviews, surveys, usability tests and turning findings into insights.", "User Research;Usability Testing;Interviews;Surveys", "Intermediate", 16, "UX fundamentals"),
        ("Interactive Prototyping and Micro-interactions", "High-fidelity prototypes with transitions and micro-interactions.", "Prototyping;Figma;Micro-interactions;Interaction Design", "Intermediate", 12, "Figma basics"),
        ("Design Systems and Component Libraries", "Tokens, reusable components and documentation for scalable design.", "Design Systems;Figma;Components;Accessibility", "Intermediate", 14, "Figma; UI principles"),
        ("Accessible and Inclusive Design", "WCAG guidelines, inclusive research and accessible interface patterns.", "Accessibility;WCAG;Inclusive Design;UX Design", "Advanced", 12, "UI design principles"),
        ("UX Portfolio and Case Study Building", "Structure case studies and present design decisions convincingly.", "Portfolio;Case Studies;UX Design;Storytelling", "Advanced", 10, "Design projects"),
    ],
}

PLATFORMS = ["Coursera", "Udemy", "edX", "NPTEL", "YouTube", "freeCodeCamp", "GeeksforGeeks"]
FORMAT_BY_PLATFORM = {
    "Coursera": "Video lectures", "edX": "Video lectures", "NPTEL": "Video lectures",
    "YouTube": "Video lectures", "Udemy": "Hands-on projects",
    "freeCodeCamp": "Hands-on projects", "Kaggle Learn": "Hands-on projects",
    "GeeksforGeeks": "Reading & documentation",
}
URL_TEMPLATES = {
    "Coursera": "https://www.coursera.org/search?query={q}",
    "Udemy": "https://www.udemy.com/courses/search/?q={q}",
    "edX": "https://www.edx.org/search?q={q}",
    "YouTube": "https://www.youtube.com/results?search_query={q}",
    "freeCodeCamp": "https://www.freecodecamp.org/news/search/?query={q}",
    "NPTEL": "https://nptel.ac.in/courses",
    "Kaggle Learn": "https://www.kaggle.com/learn",
    "GeeksforGeeks": "https://www.geeksforgeeks.org/",
}
AI_DS_CATEGORIES = {"Artificial Intelligence", "Machine Learning", "Data Science"}


def build_dataset(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, counter = [], 1
    for cat_idx, ((category, career), courses) in enumerate(CATALOG.items()):
        for i, (name, desc, skills, difficulty, hours, prereq) in enumerate(courses):
            platform = PLATFORMS[(cat_idx * 3 + i) % len(PLATFORMS)]
            if platform == "GeeksforGeeks" and category in AI_DS_CATEGORIES:
                platform = "Kaggle Learn"
            base = {"Beginner": 250_000, "Intermediate": 120_000, "Advanced": 45_000}[difficulty]
            rows.append({
                "course_id": f"C{counter:03d}",
                "course_name": name,
                "category": category,
                "description": desc,
                "skills": skills,
                "difficulty": difficulty,
                "duration_hours": hours,
                "platform": platform,
                "rating": round(float(rng.uniform(4.0, 4.9)), 1),
                "enrollments": int(base * rng.uniform(0.2, 3.0)),
                "prerequisites": prereq,
                "career_domain": career,
                "course_url": URL_TEMPLATES[platform].format(q=quote_plus(name)),
                "learning_format": FORMAT_BY_PLATFORM[platform],
                "path_order": i + 1,
                "data_source": "Sample dataset (illustrative, not official)",
            })
            counter += 1
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out_path = Path(__file__).resolve().parent / "courses.csv"
    df = build_dataset()
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} sample courses to {out_path}")
