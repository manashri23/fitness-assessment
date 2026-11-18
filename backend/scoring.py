class FitnessScorer:
    """
    Computes overall weighted Fitness Score (0-100) based on sub-metrics:
      - Strength Score (30%)
      - Flexibility Score (25%)
      - Balance Score (20%)
      - Posture Score (25%)
    """
    STRENGTH_WEIGHT = 0.30
    FLEXIBILITY_WEIGHT = 0.25
    BALANCE_WEIGHT = 0.20
    POSTURE_WEIGHT = 0.25

    @classmethod
    def calculate_fitness_score(cls, strength, flexibility, balance, posture):
        """
        Calculates composite fitness score and sub-score breakdown.
        `strength`, `flexibility`, `balance`, `posture` are floats between 0 and 100.
        """
        strength = max(0.0, min(100.0, float(strength)))
        flexibility = max(0.0, min(100.0, float(flexibility)))
        balance = max(0.0, min(100.0, float(balance)))
        posture = max(0.0, min(100.0, float(posture)))

        weighted_score = (
            (strength * cls.STRENGTH_WEIGHT) +
            (flexibility * cls.FLEXIBILITY_WEIGHT) +
            (balance * cls.BALANCE_WEIGHT) +
            (posture * cls.POSTURE_WEIGHT)
        )

        fitness_score = round(weighted_score, 1)

        # Rating Category
        if fitness_score >= 85.0:
            tier = "Excellent (Athlete Level)"
        elif fitness_score >= 70.0:
            tier = "Good (Fit)"
        elif fitness_score >= 50.0:
            tier = "Moderate (Needs Work)"
        else:
            tier = "Novice / Attention Required"

        return {
            "fitness_score": fitness_score,
            "tier": tier,
            "breakdown": {
                "strength": round(strength, 1),
                "flexibility": round(flexibility, 1),
                "balance": round(balance, 1),
                "posture": round(posture, 1)
            }
        }

    @classmethod
    def compute_strength_from_reps(cls, squats, pushups):
        """Estimate strength score (0-100) based on rep counts."""
        squats = max(0, int(squats))
        pushups = max(0, int(pushups))
        if squats == 0 and pushups == 0:
            return 0.0
        squat_score = min(100.0, (squats / 15.0) * 100.0) if squats > 0 else 0.0
        pushup_score = min(100.0, (pushups / 15.0) * 100.0) if pushups > 0 else 0.0
        if squats > 0 and pushups > 0:
            return round((squat_score * 0.5) + (pushup_score * 0.5), 1)
        elif squats > 0:
            return round(squat_score, 1)
        else:
            return round(pushup_score, 1)

    @classmethod
    def compute_exercise_strength(cls, category="gym", exercise="squat", reps=0, squats=0, pushups=0, frames_count=0):
        """
        Calculates strength score dynamically based on the exercise performed and actual rep volume.
        Directly correlates rep count with repetition strength power output.
        """
        ex = (exercise or "squat").lower()
        cat = (category or "gym").lower()
        reps = max(0, int(reps))

        # Direct squat or pushup exercises
        if ex == "squat":
            actual_reps = max(reps, squats)
            if actual_reps == 0:
                return 0.0
            return round(min(100.0, (actual_reps / 12.0) * 100.0), 1)

        elif ex == "pushup":
            actual_reps = max(reps, pushups)
            if actual_reps == 0:
                return 0.0
            return round(min(100.0, (actual_reps / 12.0) * 100.0), 1)

        elif ex in ["plank", "lunge"]:
            if reps == 0:
                return 0.0
            return round(min(100.0, (reps / 8.0) * 100.0), 1)

        elif cat == "yoga":
            # Yoga core hold endurance: each hold interval adds endurance strength
            if reps == 0:
                return 25.0 if frames_count > 10 else 0.0
            return round(min(100.0, 30.0 + (reps * 14.0)), 1)

        elif cat in ["basketball", "cricket", "athletics"]:
            actual_reps = max(reps, squats + pushups)
            if actual_reps == 0:
                return 0.0
            return round(min(100.0, (actual_reps / 8.0) * 100.0), 1)

        # Overall rep total fallback
        total_reps = max(reps, squats + pushups)
        if total_reps == 0:
            return 0.0
        return round(min(100.0, (total_reps / 12.0) * 100.0), 1)
