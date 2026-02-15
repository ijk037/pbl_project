const express = require("express");
const router = express.Router();
const facultyController = require("../controllers/facultyController");

/* ==========================
   GET STUDENT BY REG NO
========================== */
router.get("/:regNo", facultyController.getStudentByRegNo);

/* ==========================
   UPLOAD MARKS
========================== */
router.post("/upload", async (req, res) => {
  try {
    const { regNo, semester, mid, end } = req.body;

    const student = await Student.findOne({ regNo });

    if (!student) {
      return res.status(404).json({ msg: "Student not found" });
    }

    const semNumber = Number(semester);

    // 🔎 Check if semester already exists
    const existingExam = student.exams.find(
      exam => exam.semester === semNumber
    );

    if (existingExam) {
      // ✅ UPDATE existing semester
      if (mid !== undefined && mid !== "")
        existingExam.mid = Number(mid);

      if (end !== undefined && end !== "")
        existingExam.end = Number(end);
    } else {
      // ✅ ADD new semester
      student.exams.push({
        semester: semNumber,
        mid: mid ? Number(mid) : 0,
        end: end ? Number(end) : 0
      });
    }

    await student.save();

    res.json({ msg: "Marks saved successfully", student });

  } catch (err) {
    console.log(err);
    res.status(500).json({ msg: "Upload failed" });
  }
});


module.exports = router;
