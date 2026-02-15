const express = require("express");
const router = express.Router();

const {
  signupStudent,
  loginStudent,
  loginFaculty
} = require("../controllers/authController");

router.post("/student-signup", signupStudent);
router.post("/student-login", loginStudent);
router.post("/faculty-login", loginFaculty);

module.exports = router;
