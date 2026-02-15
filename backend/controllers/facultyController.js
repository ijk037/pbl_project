const Student = require("../models/Student");


// Get student by registration number
exports.getStudentByRegNo = async (req, res) => {
  try {
    const { regNo } = req.params;

    const student = await Student.findOne({ regNo });

    if (!student)
      return res.status(404).json({ msg: "Student not found" });

    res.json(student);

  } catch (err) {
    res.status(500).json({ msg: "Server error" });
  }
};


// Upload new marks
exports.uploadMarks = async (req, res) => {
  try {
    const { regNo, mid, end, semester } = req.body;

    const student = await Student.findOne({ regNo });

    if (!student)
      return res.status(404).json({ msg: "Student not found" });

    student.exams.push({
      semester: Number(semester),
      mid: Number(mid),
      end: Number(end)
    });

    await student.save();

    res.json({ msg: "Marks uploaded successfully", student });

  } catch (err) {
    res.status(500).json({ msg: "Server error" });
  }
};
