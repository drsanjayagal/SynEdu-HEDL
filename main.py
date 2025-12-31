import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import random
from typing import List, Dict, Any
import os


class SynEduHEDLGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)

        # Configuration
        self.num_students = 20000
        self.num_courses = 180
        self.num_faculty = 420
        self.num_semesters = 6
        self.target_lms_events = 120000

        # Define categorical values
        self.genders = ['Male', 'Female', 'Other']
        self.admission_types = ['Merit', 'Management', 'Transfer']
        self.programs = ['B.Tech', 'M.Tech', 'MCA']
        self.specializations = ['AI', 'DS', 'CSE', 'IT']
        self.course_types = ['Theory', 'Lab', 'Project']
        self.delivery_modes = ['Blended', 'Online', 'Offline']
        self.event_types = ['Login', 'VideoView', 'Quiz', 'Forum', 'AssignmentView', 'Download']
        self.resource_types = ['Video', 'PDF', 'Assignment', 'Quiz', 'LectureNotes', 'Code']
        self.device_types = ['Mobile', 'Laptop', 'Tablet', 'Desktop']
        self.assessment_types = ['Quiz', 'Mid', 'End', 'Assignment']
        self.submission_modes = ['LMS', 'Offline']
        self.final_grades = ['A', 'B', 'C', 'D', 'F']

    def generate_student_profiles(self):
        """Generate STUDENT_PROFILE table efficiently"""
        print("Generating student profiles...")

        data = {
            'student_id': np.arange(1, self.num_students + 1),
            'gender': np.random.choice(self.genders, size=self.num_students, p=[0.55, 0.42, 0.03]),
            'age': np.random.randint(18, 27, size=self.num_students),
            'admission_type': np.random.choice(self.admission_types, size=self.num_students, p=[0.6, 0.35, 0.05]),
            'program': np.random.choice(self.programs, size=self.num_students, p=[0.7, 0.2, 0.1]),
            'specialization': np.random.choice(self.specializations, size=self.num_students),
            'socioeconomic_index': np.random.randint(1, 6, size=self.num_students),
            'first_gen_learner': np.random.choice([True, False], size=self.num_students, p=[0.35, 0.65]),
            'baseline_digital_literacy': np.clip(np.random.beta(2, 2, size=self.num_students) + 0.1, 0, 1),
            'enrollment_year': np.random.randint(2019, 2023, size=self.num_students)
        }

        return pd.DataFrame(data)

    def generate_course_metadata(self):
        """Generate COURSE_METADATA table efficiently"""
        print("Generating course metadata...")

        data = []
        course_patterns = {
            'Theory': {'credits': [3, 4], 'difficulty_range': (2, 5), 'tech_dep_range': (2, 4)},
            'Lab': {'credits': [2, 3], 'difficulty_range': (3, 5), 'tech_dep_range': (4, 5)},
            'Project': {'credits': [4, 5], 'difficulty_range': (4, 5), 'tech_dep_range': (4, 5)}
        }

        for i in range(1, self.num_courses + 1):
            course_type = np.random.choice(self.course_types)
            pattern = course_patterns[course_type]

            # Generate assessment weightage as JSON
            if course_type == 'Theory':
                assessment_split = {"Internal": 40, "External": 60}
            elif course_type == 'Lab':
                assessment_split = {"Internal": 70, "External": 30}
            else:
                assessment_split = {"Internal": 80, "External": 20}

            data.append({
                'course_id': i,
                'course_type': course_type,
                'credit_value': np.random.choice(pattern['credits']),
                'delivery_mode': np.random.choice(self.delivery_modes, p=[0.6, 0.3, 0.1]),
                'assessment_weightage': json.dumps(assessment_split),
                'difficulty_level': np.random.randint(*pattern['difficulty_range']),
                'tech_dependency': np.random.randint(*pattern['tech_dep_range'])
            })

        return pd.DataFrame(data)

    def generate_lms_interaction_logs(self, student_profiles, course_metadata):
        """Generate LMS_INTERACTION_LOG table efficiently"""
        print("Generating LMS interaction logs...")

        # Create base data for all events
        student_ids = np.random.choice(
            student_profiles['student_id'].values,
            size=self.target_lms_events,
            replace=True
        )

        course_ids = np.random.choice(
            course_metadata['course_id'].values,
            size=self.target_lms_events,
            replace=True
        )

        # Generate event types with probabilities
        event_probs = [0.3, 0.2, 0.15, 0.15, 0.1, 0.1]
        event_types = np.random.choice(self.event_types, size=self.target_lms_events, p=event_probs)

        # Generate timestamps (spread over 6 semesters, ~90 days each)
        total_days = self.num_semesters * 90
        days_offset = np.random.randint(0, total_days, size=self.target_lms_events)
        base_date = datetime(2023, 1, 15)

        timestamps = [
            base_date + timedelta(days=int(days_offset[i]),
                                  hours=np.random.randint(8, 20),
                                  minutes=np.random.randint(0, 60))
            for i in range(self.target_lms_events)
        ]

        # Create dataframe
        data = {
            'event_id': np.arange(1, self.target_lms_events + 1),
            'student_id': student_ids,
            'course_id': course_ids,
            'event_type': event_types,
            'session_duration': np.clip(np.random.exponential(15, size=self.target_lms_events), 1, 120),
            'resource_type': np.random.choice(self.resource_types, size=self.target_lms_events),
            'timestamp': timestamps,
            'device_type': np.random.choice(self.device_types, size=self.target_lms_events, p=[0.4, 0.5, 0.05, 0.05]),
            'network_quality': np.random.randint(1, 6, size=self.target_lms_events)
        }

        return pd.DataFrame(data)

    def generate_assessment_performance(self, student_profiles, course_metadata):
        """Generate ASSESSMENT_PERFORMANCE table efficiently"""
        print("Generating assessment performance...")

        # Estimate total assessments: students * avg_courses * avg_assessments
        avg_courses_per_student = 5
        avg_assessments_per_course = 3
        total_assessments = self.num_students * avg_courses_per_student * avg_assessments_per_course

        # Create arrays for all assessments
        student_ids = np.random.choice(
            student_profiles['student_id'].values,
            size=total_assessments,
            replace=True
        )

        course_ids = np.random.choice(
            course_metadata['course_id'].values,
            size=total_assessments,
            replace=True
        )

        # Generate assessment types
        assessment_types = np.random.choice(
            self.assessment_types,
            size=total_assessments,
            p=[0.4, 0.3, 0.2, 0.1]
        )

        # Generate max marks based on assessment type
        max_marks = np.zeros(total_assessments)
        for i, a_type in enumerate(assessment_types):
            if a_type == 'Quiz':
                max_marks[i] = np.random.choice([10, 15, 20])
            elif a_type == 'Mid':
                max_marks[i] = np.random.choice([50, 60])
            elif a_type == 'End':
                max_marks[i] = np.random.choice([70, 80, 100])
            else:  # Assignment
                max_marks[i] = np.random.choice([20, 25, 30])

        # Create lookup dictionaries for student and course attributes
        student_lookup = student_profiles.set_index('student_id')
        course_lookup = course_metadata.set_index('course_id')

        # Calculate performance scores
        performance_scores = np.zeros(total_assessments)
        for i in range(total_assessments):
            student_id = student_ids[i]
            if student_id in student_lookup.index:
                student = student_lookup.loc[student_id]
                base_perf = student['baseline_digital_literacy'] * 0.3
                base_perf += (student['socioeconomic_index'] / 5) * 0.2
                random_comp = np.random.normal(0.5, 0.2)
                performance_scores[i] = np.clip(base_perf + random_comp, 0.2, 1.0)
            else:
                performance_scores[i] = np.clip(np.random.normal(0.6, 0.2), 0.2, 1.0)

        obtained_marks = np.round(performance_scores * max_marks, 2)

        # Create submission modes and plagiarism flags
        submission_modes = np.random.choice(
            self.submission_modes,
            size=total_assessments,
            p=[0.8, 0.2]
        )

        plagiarism_flags = np.random.choice(
            [True, False],
            size=total_assessments,
            p=[0.05, 0.95]
        )

        # Create dataframe
        data = {
            'assessment_id': np.arange(1, total_assessments + 1),
            'student_id': student_ids,
            'course_id': course_ids,
            'assessment_type': assessment_types,
            'max_marks': max_marks.astype(int),
            'obtained_marks': obtained_marks,
            'submission_mode': submission_modes,
            'plagiarism_flag': plagiarism_flags
        }

        return pd.DataFrame(data)

    def generate_engagement_metrics(self, lms_logs, assessment_performance):
        """Generate ENGAGEMENT_METRICS table efficiently"""
        print("Generating engagement metrics...")

        # Get unique student-course pairs from LMS logs
        unique_pairs = lms_logs[['student_id', 'course_id']].drop_duplicates()

        # Initialize lists for metrics
        weekly_login_freq = []
        avg_session_times = []
        content_completion = []
        forum_scores = []

        # Process each pair
        for _, row in unique_pairs.iterrows():
            student_id = row['student_id']
            course_id = row['course_id']

            # Filter logs for this student-course pair
            mask = (lms_logs['student_id'] == student_id) & (lms_logs['course_id'] == course_id)
            pair_logs = lms_logs[mask]

            # Weekly login frequency
            if len(pair_logs) > 0:
                login_events = pair_logs[pair_logs['event_type'] == 'Login']
                if len(login_events) > 0:
                    time_range = (login_events['timestamp'].max() - login_events['timestamp'].min()).days
                    weeks = max(1, time_range / 7)
                    weekly_login_freq.append(len(login_events) / weeks)
                else:
                    weekly_login_freq.append(0)
            else:
                weekly_login_freq.append(0)

            # Average session time
            if len(pair_logs) > 0:
                avg_session_times.append(pair_logs['session_duration'].mean())
            else:
                avg_session_times.append(0)

            # Content completion rate (simulated)
            content_completion.append(np.clip(np.random.beta(2, 2) * 0.8 + 0.1, 0, 1))

            # Forum participation score
            if len(pair_logs) > 0:
                forum_count = len(pair_logs[pair_logs['event_type'] == 'Forum'])
                forum_scores.append(min(forum_count / 10, 1))
            else:
                forum_scores.append(0)

        # Calculate assessment timeliness using efficient aggregation
        if not assessment_performance.empty:
            # Use agg instead of apply to avoid FutureWarning
            assessment_summary = assessment_performance.groupby(['student_id', 'course_id']).agg(
                total_obtained=('obtained_marks', 'sum'),
                total_max=('max_marks', 'sum')
            ).reset_index()

            # Merge with unique pairs
            merged = pd.merge(
                unique_pairs,
                assessment_summary,
                on=['student_id', 'course_id'],
                how='left'
            )

            # Calculate average scores
            merged['avg_score'] = merged.apply(
                lambda row: row['total_obtained'] / row['total_max']
                if pd.notnull(row['total_max']) and row['total_max'] > 0
                else 0.5,
                axis=1
            )

            # Calculate timeliness
            timeliness = np.clip(merged['avg_score'].values * 0.8 + np.random.normal(0.1, 0.05, len(merged)), 0, 1)
        else:
            timeliness = np.clip(np.random.normal(0.5, 0.2, len(unique_pairs)), 0, 1)

        # Create final dataframe
        engagement_df = pd.DataFrame({
            'student_id': unique_pairs['student_id'].values,
            'course_id': unique_pairs['course_id'].values,
            'weekly_login_frequency': np.round(weekly_login_freq, 2),
            'average_session_time': np.round(avg_session_times, 2),
            'content_completion_rate': np.round(content_completion, 2),
            'forum_participation_score': np.round(forum_scores, 2),
            'assessment_timeliness': np.round(timeliness, 2)
        })

        return engagement_df

    def generate_outcome_labels(self, student_profiles, course_metadata, engagement_metrics, assessment_performance):
        """Generate OUTCOME_LABELS table efficiently"""
        print("Generating outcome labels...")

        # Get unique student-course pairs from engagement metrics
        pairs = engagement_metrics[['student_id', 'course_id']].drop_duplicates()

        # Create lookup dictionaries for faster access
        student_dict = student_profiles.set_index('student_id').to_dict('index')
        course_dict = course_metadata.set_index('course_id').to_dict('index')
        engagement_dict = engagement_metrics.set_index(['student_id', 'course_id']).to_dict('index')

        # Pre-calculate assessment performance
        if not assessment_performance.empty:
            assessment_avg = assessment_performance.groupby(['student_id', 'course_id']).agg(
                total_obtained=('obtained_marks', 'sum'),
                total_max=('max_marks', 'sum')
            )
            assessment_avg['avg_performance'] = assessment_avg['total_obtained'] / assessment_avg['total_max']
            assessment_avg = assessment_avg['avg_performance'].to_dict()
        else:
            assessment_avg = {}

        outcomes = []

        for _, row in pairs.iterrows():
            student_id = row['student_id']
            course_id = row['course_id']

            # Get data using dictionaries
            student = student_dict.get(student_id)
            course = course_dict.get(course_id)

            if not student or not course:
                continue

            # Get engagement metrics
            engagement_key = (student_id, course_id)
            engagement = engagement_dict.get(engagement_key, {})

            # Get assessment performance
            avg_performance = assessment_avg.get(engagement_key, 0.5)

            # Calculate engagement score
            engagement_score = engagement.get('content_completion_rate', 0.5) * 0.3
            engagement_score += engagement.get('assessment_timeliness', 0.5) * 0.2
            engagement_score += engagement.get('forum_participation_score', 0.5) * 0.1

            # Student factors
            student_factor = student['baseline_digital_literacy'] * 0.2
            student_factor += (student['socioeconomic_index'] / 5) * 0.1

            # Course difficulty adjustment
            difficulty_factor = 1 - (course['difficulty_level'] / 10)

            # Calculate final score
            final_score = (avg_performance * 0.4 + engagement_score * 0.4 + student_factor * 0.2) * difficulty_factor

            # Determine final grade
            if final_score >= 0.85:
                final_grade = 'A'
            elif final_score >= 0.75:
                final_grade = 'B'
            elif final_score >= 0.65:
                final_grade = 'C'
            elif final_score >= 0.50:
                final_grade = 'D'
            else:
                final_grade = 'F'

            # Learning gain index
            learning_gain = np.clip(final_score * 0.8 + np.random.normal(0.1, 0.05), 0, 1)

            # Dropout risk
            dropout_risk = final_grade in ['D', 'F'] or engagement.get('content_completion_rate', 0.5) < 0.4

            # Course satisfaction
            satisfaction = int(np.clip(final_score * 5, 1, 5))

            outcomes.append({
                'student_id': student_id,
                'course_id': int(course_id),
                'final_grade': final_grade,
                'learning_gain_index': round(learning_gain, 3),
                'dropout_risk': dropout_risk,
                'course_satisfaction': satisfaction
            })

        return pd.DataFrame(outcomes)

    def generate_full_dataset(self):
        """Generate complete dataset"""
        print("=" * 60)
        print("Generating SynEdu-HEDL Dataset")
        print("=" * 60)

        # Generate base tables
        student_profiles = self.generate_student_profiles()
        course_metadata = self.generate_course_metadata()
        lms_logs = self.generate_lms_interaction_logs(student_profiles, course_metadata)
        assessment_performance = self.generate_assessment_performance(student_profiles, course_metadata)

        # Generate derived tables
        engagement_metrics = self.generate_engagement_metrics(lms_logs, assessment_performance)
        outcome_labels = self.generate_outcome_labels(
            student_profiles, course_metadata, engagement_metrics, assessment_performance
        )

        datasets = {
            'STUDENT_PROFILE': student_profiles,
            'COURSE_METADATA': course_metadata,
            'LMS_INTERACTION_LOG': lms_logs,
            'ASSESSMENT_PERFORMANCE': assessment_performance,
            'ENGAGEMENT_METRICS': engagement_metrics,
            'OUTCOME_LABELS': outcome_labels
        }

        # Print summary
        print("\n" + "=" * 60)
        print("Dataset Summary")
        print("=" * 60)
        for name, df in datasets.items():
            print(f"{name}: {len(df):,} records")

        return datasets

    def save_datasets(self, datasets, output_dir='synedu_hedl_data'):
        """Save all datasets to CSV files"""
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nSaving datasets to '{output_dir}' directory...")
        for name, df in datasets.items():
            filename = f"{output_dir}/{name}.csv"
            df.to_csv(filename, index=False)
            print(f"  ✓ {filename} ({len(df):,} records)")

        # Save data dictionary
        self.save_data_dictionary(datasets, output_dir)

    def save_data_dictionary(self, datasets, output_dir):
        """Save data dictionary as JSON"""
        data_dict = {
            'dataset_name': 'SynEdu-HEDL: A Synthetic Higher Education Digital Learning Dataset',
            'purpose': 'To simulate realistic learner behavior, academic performance, and digital engagement patterns in higher education for privacy-preserving learning analytics and educational decision support research.',
            'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_scale': {
                'students': len(datasets['STUDENT_PROFILE']),
                'courses': len(datasets['COURSE_METADATA']),
                'faculty': self.num_faculty,
                'semesters': self.num_semesters,
                'lms_interaction_events': len(datasets['LMS_INTERACTION_LOG'])
            },
            'tables': {}
        }

        # Add table schemas
        for name, df in datasets.items():
            data_dict['tables'][name] = {
                'record_count': len(df),
                'columns': []
            }

            for col in df.columns:
                dtype = str(df[col].dtype)
                sample_value = df[col].iloc[0] if len(df) > 0 else None
                data_dict['tables'][name]['columns'].append({
                    'name': col,
                    'dtype': dtype,
                    'sample_value': str(sample_value)[:100] if sample_value is not None else None
                })

        # Save to file
        dict_file = os.path.join(output_dir, 'data_dictionary.json')
        with open(dict_file, 'w') as f:
            json.dump(data_dict, f, indent=2, default=str)

        print(f"  ✓ {dict_file}")


def main():
    """Main execution function"""
    print("SynEdu-HEDL Dataset Generator")
    print("=" * 60)

    # Ask for configuration
    print("\nConfiguration Options:")
    print("1. Full dataset (20,000 students, 120,000+ LMS events)")
    print("2. Test dataset (2,000 students, 20,000 LMS events)")

    choice = input("\nSelect option (1 or 2): ").strip()

    # Create generator
    generator = SynEduHEDLGenerator(seed=42)

    if choice == '2':
        # Test mode
        print("\nRunning in TEST MODE...")
        generator.num_students = 2000
        generator.target_lms_events = 20000
    else:
        print("\nRunning in FULL MODE...")

    # Generate dataset
    try:
        datasets = generator.generate_full_dataset()

        # Ask to save
        save = input("\nSave datasets to CSV files? (y/n): ").lower().strip()
        if save == 'y':
            output_dir = input("Enter output directory (default: 'synedu_hedl_data'): ").strip()
            if not output_dir:
                output_dir = 'synedu_hedl_data'

            generator.save_datasets(datasets, output_dir)

        # Show sample
        show_sample = input("\nShow sample data from each table? (y/n): ").lower().strip()
        if show_sample == 'y':
            print("\n" + "=" * 60)
            print("Sample Data (first 2 rows)")
            print("=" * 60)

            for name, df in datasets.items():
                print(f"\n{name}:")
                print(df.head(2).to_string())
                print("-" * 40)

        print("\n✅ Dataset generation completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during dataset generation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()