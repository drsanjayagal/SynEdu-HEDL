# SynEdu-HEDL: Synthetic Higher Education Digital Learning Dataset

## 📊 Dataset Overview

**SynEdu-HEDL** is a comprehensive synthetic dataset designed to simulate realistic learner behavior, academic performance, and digital engagement patterns in higher education. This dataset enables privacy-preserving research in learning analytics, educational data mining, and decision support systems without compromising real student data.

### Key Features
- **Privacy-Preserving**: 100% synthetic data with no real student information
- **Realistic Correlations**: Patterns mirror actual educational environments
- **Multi-Table Structure**: 6 interconnected tables with referential integrity
- **Scalable Design**: Configurable to generate datasets of various sizes
- **Research-Ready**: Includes derived metrics and outcome labels for ML/Analytics

## 🎯 Purpose

This dataset supports research in:
- Learning Analytics and Educational Data Mining
- Predictive Modeling of Student Performance
- Digital Engagement Pattern Analysis
- Educational Intervention Effectiveness
- Privacy-Preserving Educational Research
- Algorithm Development for Learning Management Systems

## 📈 Dataset Scale

| Component | Count | Description |
|-----------|-------|-------------|
| Students | 20,000 | Synthetic learners with demographic profiles |
| Courses | 180 | Various course types and difficulty levels |
| Faculty | 420 | Teaching staff (implied, not explicitly modeled) |
| Semesters | 6 | Academic periods spanning 3 years |
| LMS Events | 120,000+ | Digital interaction records |
| Assessments | ~300,000 | Performance records across courses |

## 🏫 Institutions Modeled
- Engineering universities
- Multidisciplinary higher education institutions
- Similar to Indian private/state universities

## 📁 Dataset Structure

### 1️⃣ **STUDENT_PROFILE** Table
Contains demographic and background information for synthetic learners.

| Attribute | Type | Description | Values/Range |
|-----------|------|-------------|--------------|
| student_id | Integer | Unique synthetic learner ID | 1-20,000 |
| gender | Categorical | Gender identity | Male/Female/Other |
| age | Integer | Student age | 18-26 |
| admission_type | Categorical | Admission category | Merit/Management/Transfer |
| program | Categorical | Degree program | B.Tech/M.Tech/MCA |
| specialization | Categorical | Academic specialization | AI, DS, CSE, IT |
| socioeconomic_index | Integer | Proxy for access/resources | 1-5 (Low-High) |
| first_gen_learner | Boolean | First-generation student | True/False |
| baseline_digital_literacy | Float | Initial tech readiness | 0-1 |
| enrollment_year | Integer | Admission year | 2019-2022 |

### 2️⃣ **COURSE_METADATA** Table
Contains course characteristics and metadata.

| Attribute | Type | Description | Values/Range |
|-----------|------|-------------|--------------|
| course_id | Integer | Unique course identifier | 1-180 |
| course_type | Categorical | Course format | Theory/Lab/Project |
| credit_value | Integer | Academic credits | 2-5 |
| delivery_mode | Categorical | Instructional delivery | Blended/Online/Offline |
| assessment_weightage | JSON | Internal/External split | {"Internal": X, "External": Y} |
| difficulty_level | Integer | Academic rigor | 1-5 (Easy-Hard) |
| tech_dependency | Integer | LMS reliance | 1-5 (Low-High) |

### 3️⃣ **LMS_INTERACTION_LOG** Table
Detailed Learning Management System interaction records.

| Attribute | Type | Description | Values/Range |
|-----------|------|-------------|--------------|
| event_id | Integer | Unique log entry | 1-120,000+ |
| student_id | Integer | Foreign key → Student | 1-20,000 |
| course_id | Integer | Foreign key → Course | 1-180 |
| event_type | Categorical | Interaction type | Login, VideoView, Quiz, Forum, etc. |
| session_duration | Float | Time spent (minutes) | 1-120 |
| resource_type | Categorical | Content type | Video, PDF, Assignment, etc. |
| timestamp | Datetime | Interaction time | 2023-2025 |
| device_type | Categorical | Access device | Mobile/Laptop/Tablet/Desktop |
| network_quality | Integer | Connectivity proxy | 1-5 (Poor-Excellent) |

### 4️⃣ **ASSESSMENT_PERFORMANCE** Table
Student assessment scores and submission details.

| Attribute | Type | Description | Values/Range |
|-----------|------|-------------|--------------|
| assessment_id | Integer | Unique assessment | 1-300,000+ |
| student_id | Integer | Foreign key | 1-20,000 |
| course_id | Integer | Foreign key | 1-180 |
| assessment_type | Categorical | Assessment category | Quiz/Mid/End/Assignment |
| max_marks | Integer | Total marks | Varies by type |
| obtained_marks | Float | Student score | 0-max_marks |
| submission_mode | Categorical | Submission method | LMS/Offline |
| plagiarism_flag | Boolean | Academic integrity | True/False |

### 5️⃣ **ENGAGEMENT_METRICS** Table
Derived engagement indicators from interaction logs.

| Metric | Type | Description | Range |
|--------|------|-------------|-------|
| weekly_login_frequency | Float | LMS usage pattern | 0-20 |
| average_session_time | Float | Engagement depth | 0-120 min |
| content_completion_rate | Float | Learning persistence | 0-1 |
| forum_participation_score | Float | Collaborative learning | 0-1 |
| assessment_timeliness | Float | Self-regulation | 0-1 |

### 6️⃣ **OUTCOME_LABELS** Table
Learning outcomes and predictive labels for analytics.

| Attribute | Type | Description | Values/Range |
|-----------|------|-------------|--------------|
| final_grade | Categorical | Course final grade | A/B/C/D/F |
| learning_gain_index | Float | Pre-Post improvement | 0-1 |
| dropout_risk | Binary | At-risk identification | True/False |
| course_satisfaction | Integer | Student perception | 1-5 (Low-High) |

## 🔧 Synthetic Data Generation Logic

The dataset was generated using a hybrid rule-based and probabilistic modeling framework:

```python
Key Generation Principles:
1. Student engagement patterns are conditioned on:
   - Digital literacy levels
   - Course difficulty
   - Delivery mode (Blended/Online/Offline)
   
2. Assessment outcomes are probabilistically influenced by:
   - Engagement metrics
   - Attendance regularity (derived from LMS logs)
   - Prior academic performance
   
3. Realistic correlations observed in higher education:
   - Socioeconomic status ↔ Resource access
   - Digital literacy ↔ LMS engagement
   - Engagement metrics ↔ Academic performance
   - Course difficulty ↔ Dropout risk
