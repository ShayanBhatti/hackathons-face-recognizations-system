from datetime import date
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..services.supabase_service import db_request

router = APIRouter(tags=["Students and Attendance"])


class StudentCreate(BaseModel):
    student_code: str = Field(min_length=1, max_length=50)
    full_name: str = Field(min_length=1, max_length=150)
    email: Optional[str] = None
    department: Optional[str] = None


class AttendanceMark(BaseModel):
    student_id: int = Field(gt=0)
    status: str = Field(default="present", pattern="^(present|late)$")
    confidence: Optional[float] = Field(default=None, ge=0, le=1)


@router.post("/students", status_code=201)
def create_student(student: StudentCreate):
    data = student.model_dump()
    data["student_code"] = data["student_code"].strip()
    data["full_name"] = data["full_name"].strip()

    if not data["student_code"] or not data["full_name"]:
        raise HTTPException(422, "Student code and name cannot be blank")

    if data["email"]:
        data["email"] = data["email"].strip().lower() or None

    if data["department"]:
        data["department"] = data["department"].strip() or None

    try:
        rows = db_request("POST", "students", payload=data)
        return rows[0] if rows else {"message": "Student created"}
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    except RuntimeError as exc:
        raise HTTPException(502, str(exc))


@router.get("/students")
def list_students(
    active_only: bool = True,
    search: Optional[str] = Query(default=None, max_length=100),
):
    params = {
        "select": "*",
        "order": "created_at.desc",
    }
    if active_only:
        params["is_active"] = "eq.true"

    if search and search.strip():
        term = search.strip().replace(",", " ")
        params["or"] = (
            f"(full_name.ilike.*{term}*,student_code.ilike.*{term}*)"
        )

    try:
        return db_request("GET", "students", params=params)
    except RuntimeError as exc:
        raise HTTPException(502, str(exc))


@router.post("/attendance/mark")
def mark_attendance(record: AttendanceMark):
    today = date.today().isoformat()

    try:
        students = db_request(
            "GET",
            "students",
            params={
                "select": "id,student_code,full_name,is_active",
                "id": f"eq.{record.student_id}",
                "limit": "1",
            },
        )

        if not students:
            raise HTTPException(404, "Student not found")

        student = students[0]
        if not student["is_active"]:
            raise HTTPException(400, "This student is inactive")

        existing = db_request(
            "GET",
            "attendance",
            params={
                "select": "*",
                "student_id": f"eq.{record.student_id}",
                "attendance_date": f"eq.{today}",
                "limit": "1",
            },
        )

        if existing:
            return {
                "message": "Attendance already marked for today",
                "already_marked": True,
                "attendance": existing[0],
            }

        payload = {
            "student_id": record.student_id,
            "attendance_date": today,
            "status": record.status,
            "method": "manual",
            "confidence": record.confidence,
        }

        rows = db_request("POST", "attendance", payload=payload)
        return {
            "message": "Attendance marked successfully",
            "already_marked": False,
            "attendance": rows[0] if rows else payload,
        }

    except HTTPException:
        raise
    except ValueError as exc:
        # The database unique constraint also protects against
        # simultaneous duplicate attendance requests.
        raise HTTPException(409, str(exc))
    except RuntimeError as exc:
        raise HTTPException(502, str(exc))


@router.get("/attendance")
def list_attendance(
    attendance_date: Optional[date] = None,
    student_id: Optional[int] = Query(default=None, gt=0),
):
    selected_date = (attendance_date or date.today()).isoformat()
    params = {
        "select": "*,students(student_code,full_name,department)",
        "attendance_date": f"eq.{selected_date}",
        "order": "marked_at.desc",
    }
    if student_id is not None:
        params["student_id"] = f"eq.{student_id}"

    try:
        return db_request("GET", "attendance", params=params)
    except RuntimeError as exc:
        raise HTTPException(502, str(exc))
