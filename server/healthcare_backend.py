#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Mock backend data for the HealthLine voice agent demo.

Swap PATIENTS and PROVIDERS with real DB/API calls in bot-healthcare.py
when going to production. Phone numbers should be stored in E.164 format.
"""

from datetime import date, timedelta

# Known patients keyed by E.164 phone number
PATIENTS = {
    "+14155551001": {
        "name": "Sarah Johnson",
        "mrn": "MRN-4821",
        "dob": "1978-03-14",
        "area": "cardiology",
        "provider": "Dr. Patel",
        "medications": [
            {
                "name": "Lisinopril",
                "dose": "10mg",
                "frequency": "once daily",
                "last_taken": str(date.today()),
                "refills_remaining": 3,
            },
            {
                "name": "Atorvastatin",
                "dose": "20mg",
                "frequency": "once daily at bedtime",
                "last_taken": str(date.today() - timedelta(days=1)),
                "refills_remaining": 1,
            },
        ],
        "appointments": [
            {
                "date": str(date.today() + timedelta(days=7)),
                "time": "10:30 AM",
                "provider": "Dr. Patel",
                "department": "Cardiology",
                "type": "Follow-up",
            }
        ],
        "balance_due": 45.00,
    },
    "+14155551002": {
        "name": "Marcus Williams",
        "mrn": "MRN-3307",
        "dob": "1965-11-22",
        "area": "primary care",
        "provider": "Dr. Chen",
        "medications": [
            {
                "name": "Metformin",
                "dose": "500mg",
                "frequency": "twice daily with meals",
                "last_taken": str(date.today()),
                "refills_remaining": 0,
            },
        ],
        "appointments": [],
        "balance_due": 0.00,
    },
    "+14155551003": {
        "name": "Elena Rodriguez",
        "mrn": "MRN-7754",
        "dob": "1990-07-08",
        "area": "orthopedics",
        "provider": "Dr. Kim",
        "medications": [
            {
                "name": "Ibuprofen",
                "dose": "400mg",
                "frequency": "as needed",
                "last_taken": str(date.today() - timedelta(days=2)),
                "refills_remaining": 2,
            },
        ],
        "appointments": [
            {
                "date": str(date.today() + timedelta(days=14)),
                "time": "2:00 PM",
                "provider": "Dr. Kim",
                "department": "Orthopedics",
                "type": "Post-op check",
            }
        ],
        "balance_due": 120.00,
    },
}

# Departments / service areas available for routing
DEPARTMENTS = {
    "cardiology": {
        "display_name": "Cardiology",
        "nurse_line": "+18005550101",
        "hours": "Monday–Friday, 8 AM to 5 PM",
    },
    "primary care": {
        "display_name": "Primary Care",
        "nurse_line": "+18005550102",
        "hours": "Monday–Friday, 7 AM to 6 PM",
    },
    "orthopedics": {
        "display_name": "Orthopedics",
        "nurse_line": "+18005550103",
        "hours": "Monday–Friday, 8 AM to 4 PM",
    },
    "mental health": {
        "display_name": "Mental Health Services",
        "nurse_line": "+18005550104",
        "hours": "Monday–Saturday, 8 AM to 8 PM",
    },
    "lab": {
        "display_name": "Laboratory / Test Results",
        "nurse_line": "+18005550105",
        "hours": "Monday–Friday, 7 AM to 5 PM",
    },
    "billing": {
        "display_name": "Billing & Insurance",
        "nurse_line": "+18005550106",
        "hours": "Monday–Friday, 9 AM to 4 PM",
    },
    "general": {
        "display_name": "General Patient Services",
        "nurse_line": "+18005550100",
        "hours": "24 hours, 7 days a week",
    },
}
