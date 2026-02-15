const Student = require("../models/Student");
const Faculty = require("../models/Faculty");
const bcrypt = require("bcryptjs");

/* ---------- STUDENT SIGNUP ---------- */
exports.signupStudent = async (req, res) => {
  try {
    const { name, email, password, regNo } = req.body;

    const existingStudent = await Student.findOne({ email });
    if (existingStudent) {
      return res.json({ success: false, msg: "Student already exists" });
    }

    const hashedPassword = await bcrypt.hash(password, 12);

    const newStudent = new Student({
      name,
      email,
      password: hashedPassword,
      regNo,
      performances: []
    });

    await newStudent.save();

    res.json({ success: true, msg: "Signup successful" });
  } catch (err) {
    console.log(err);
    res.status(500).json({ success: false, msg: "Error signing up" });
  }
};


/* ---------- STUDENT LOGIN ---------- */
exports.loginStudent = async (req, res) => {
  const { email, password } = req.body;

  const student = await Student.findOne({ email });
  if (!student)
    return res.json({ success: false, msg: "Invalid credentials" });

  const match = await bcrypt.compare(password, student.password);
  if (!match)
    return res.json({ success: false, msg: "Invalid credentials" });

  return res.json({ success: true, msg: "Login successful" });
};


/* ---------- FACULTY LOGIN ---------- */
exports.loginFaculty = async (req, res) => {

  const { email, password } = req.body;

  const faculty = await Faculty.findOne({ email });

  if (!faculty) {
    return res.json({ success: false, msg: "Invalid credentials" });
  }

  // TEMPORARY SIMPLE PASSWORD CHECK
  if (password !== "1234") {
    return res.json({ success: false, msg: "Invalid credentials" });
  }

  return res.json({ success: true, msg: "Login successful" });
};
