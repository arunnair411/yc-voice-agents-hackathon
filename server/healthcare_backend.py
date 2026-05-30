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

# ---------------------------------------------------------------------------
# Knowledge base for RAG retrieval.
#
# These are GENERAL, NON-DIAGNOSTIC informational entries: clinic policies,
# appointment prep, medication general guidance, insurance/billing FAQ. The
# voice agent retrieves over these so it can answer common patient questions
# with authoritative shop-specific info instead of hallucinating. Anything
# clinical (diagnosis, dosing changes, symptom evaluation) must still be
# routed to a registered nurse — never answered from this KB alone.
#
# Swap this list for a real document store / vector DB in production.
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = [
    {
        "title": "Appointment cancellation and rescheduling policy",
        "category": "appointments",
        "content": (
            "Appointments can be cancelled or rescheduled up to 24 hours in "
            "advance at no charge. Cancellations within 24 hours, or missed "
            "appointments, may incur a 25 dollar no-show fee. You can reschedule "
            "by phone or through the patient portal."
        ),
    },
    {
        "title": "What to bring to your appointment",
        "category": "appointments",
        "content": (
            "Please bring a photo ID, your insurance card, a list of your current "
            "medications and dosages, and any relevant medical records or referral "
            "paperwork. Arrive 15 minutes early to complete check-in."
        ),
    },
    {
        "title": "Prescription refill processing time",
        "category": "prescriptions",
        "content": (
            "Routine prescription refills are processed within 24 to 48 hours. "
            "If your prescription has no refills remaining, it must be approved by "
            "your provider, which can take up to 3 business days. Controlled "
            "substances require an in-person or telehealth visit before refill."
        ),
    },
    {
        "title": "Taking medications with food",
        "category": "medications",
        "content": (
            "General guidance: some medications are best taken with food to reduce "
            "stomach upset, while others should be taken on an empty stomach. "
            "Always follow the directions on your prescription label. For specific "
            "questions about your medication, a registered nurse or pharmacist can help."
        ),
    },
    {
        "title": "Missed medication dose",
        "category": "medications",
        "content": (
            "General guidance: if you miss a dose, take it as soon as you remember "
            "unless it is almost time for your next dose. Do not double up to make "
            "up for a missed dose. For medication-specific concerns, please speak "
            "with a registered nurse."
        ),
    },
    {
        "title": "Insurance plans accepted",
        "category": "billing",
        "content": (
            "We accept most major insurance plans including Medicare, Medicaid, "
            "Blue Cross Blue Shield, Aetna, Cigna, and UnitedHealthcare. Coverage "
            "for specific services varies by plan. Contact our billing department "
            "to verify your coverage before a visit."
        ),
    },
    {
        "title": "Paying your bill",
        "category": "billing",
        "content": (
            "Bills can be paid online through the patient portal, by phone with the "
            "billing department, or by mail. We offer interest-free payment plans "
            "for balances over 200 dollars. Financial assistance may be available "
            "for qualifying patients."
        ),
    },
    {
        "title": "Accessing lab and test results",
        "category": "lab",
        "content": (
            "Most lab results are available in the patient portal within 2 to 5 "
            "business days. Your provider will contact you about any results that "
            "need follow-up. For urgent results, call the lab line during business hours."
        ),
    },
    {
        "title": "Telehealth and virtual visits",
        "category": "appointments",
        "content": (
            "Virtual visits are available for many primary care and follow-up "
            "appointments. You will need a smartphone, tablet, or computer with a "
            "camera. A link is sent to your portal before the visit. Not all "
            "services are eligible for telehealth."
        ),
    },
    {
        "title": "Clinic hours and after-hours care",
        "category": "general",
        "content": (
            "The main clinic is open Monday through Friday. For urgent but "
            "non-emergency needs after hours, our nurse advice line is available "
            "24 hours a day. For any life-threatening emergency, call 9-1-1."
        ),
    },
]
