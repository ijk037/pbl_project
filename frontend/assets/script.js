const API = "http://localhost:5000/api";

let studentData = null;
let chart = null;
let selectedStudent = null;
let facultyChart = null;

/* ===============================
   LOAD STUDENT DASHBOARD
================================= */
window.addEventListener("DOMContentLoaded", () => {

  const email = localStorage.getItem("studentEmail");
  if (!email) return;

  fetch(`${API}/student/${email}`)
    .then(res => res.json())
    .then(student => {

      if (!student || !student.exams || student.exams.length === 0) {
        console.log("No exam data found");
        return;
      }

      studentData = student;

      // Sort by semester
      student.exams.sort((a, b) => a.semester - b.semester);

      renderDashboard(student);

    })
    .catch(err => console.log(err));
});


/* ===============================
   RENDER STUDENT DASHBOARD
================================= */
function renderDashboard(student) {

  const tableBody = document.querySelector("#marksTable tbody");
  if (!tableBody) return;

  tableBody.innerHTML = "";

  const mids = [];
  const ends = [];
  const labels = [];

  student.exams.forEach(exam => {

    labels.push("Sem " + exam.semester);
    mids.push(exam.mid);
    ends.push(exam.end);

    tableBody.innerHTML += `
      <tr>
        <td>${exam.semester}</td>
        <td>${exam.mid}</td>
        <td>${exam.end}</td>
      </tr>
    `;
  });

  document.getElementById("mid").innerText = average(mids);
  document.getElementById("end").innerText = average(ends);

  renderChart(labels, mids, ends);
}


/* ===============================
   LINE CHART
================================= */
function renderChart(labels, mids, ends) {

  if (chart) chart.destroy();

  chart = new Chart(
    document.getElementById("performanceChart"),
    {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Mid Sem",
            data: mids,
            borderColor: "#667eea",
            tension: 0.3
          },
          {
            label: "End Sem",
            data: ends,
            borderColor: "#28a745",
            tension: 0.3
          }
        ]
      },
      options: {
        responsive: true
      }
    }
  );
}


/* ===============================
   PREDICTION LOGIC
================================= */
function predictMarks() {

  if (!studentData || !studentData.exams.length) return;

  const type = document.getElementById("predictType").value;
  const exams = studentData.exams;

  let prediction = 0;

  if (type === "mid") {

    // Use last semester end exam
    const lastEnd = exams[exams.length - 1].end;
    prediction = (lastEnd * 0.8).toFixed(1);

  } else {

    if (exams.length === 1) {
      prediction = (exams[0].mid * 1.1).toFixed(1);
    } else {
      const ends = exams.map(p => p.end);
      prediction = average(ends);
    }
  }

  document.getElementById("predictionResult").innerText = prediction;
}


/* ===============================
   DOWNLOAD PDF
================================= */
function downloadPDF() {

  html2canvas(document.querySelector(".dashboard-container"))
    .then(canvas => {

      const img = canvas.toDataURL("image/png");
      const pdf = new jspdf.jsPDF("p", "mm", "a4");

      pdf.addImage(img, "PNG", 5, 5, 200, 0);
      pdf.save("Academic_Report.pdf");
    });
}


/* ===============================
   AVERAGE HELPER
================================= */
function average(arr) {
  if (!arr.length) return 0;
  return (arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(1);
}


/* ===============================
   STUDENT LOGIN
================================= */
function loginStudent() {

  const email = document.getElementById("email").value;
  const password = document.getElementById("password").value;

  fetch(`${API}/auth/student-login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password })
  })
  .then(res => res.json())
  .then(data => {

    if (data.success) {
      localStorage.setItem("studentEmail", email);
      window.location.href = "studentDashboard.html";
    } else {
      document.getElementById("message").innerText = data.msg;
    }

  })
  .catch(() => {
    document.getElementById("message").innerText = "Server not reachable";
  });
}


/* ===============================
   STUDENT SIGNUP
================================= */
function signupStudent() {

  const name = document.getElementById("name").value;
  const email = document.getElementById("email").value;
  const password = document.getElementById("password").value;
  const regNo = document.getElementById("regNo").value;

  fetch(`${API}/auth/student-signup`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, email, password, regNo })
  })
  .then(res => res.json())
  .then(data => {

    if (data.success) {
      alert("Signup successful!");
      window.location.href = "studentLogin.html";
    } else {
      document.getElementById("message").innerText = data.msg;
    }

  })
  .catch(() => {
    document.getElementById("message").innerText = "Server not reachable";
  });
}


/* ===============================
   FACULTY FETCH STUDENT
================================= */
function fetchStudentByReg() {

  const regNo = document.getElementById("searchRegNo").value;
  if (!regNo) return;

  fetch(`${API}/faculty/${regNo}`)
    .then(res => res.json())
    .then(data => {

      if (!data || !data.exams) {
        alert("Student not found");
        return;
      }

      selectedStudent = data;

      document.getElementById("studentSection").style.display = "block";
      document.getElementById("studentName").innerText = data.name;
      document.getElementById("studentEmail").innerText = data.email;
      document.getElementById("studentReg").innerText = data.regNo;

      renderFacultyTable(data.exams);
      renderFacultyAnalytics(data.exams);
    });
}


/* ===============================
   FACULTY TABLE
================================= */
function renderFacultyTable(exams) {

  const tbody = document.querySelector("#facultyTable tbody");
  if (!tbody) return;

  tbody.innerHTML = "";

  exams.forEach(exam => {
    tbody.innerHTML += `
      <tr>
        <td>${exam.semester}</td>
        <td>${exam.mid}</td>
        <td>${exam.end}</td>
      </tr>
    `;
  });
}


/* ===============================
   UPLOAD MARKS
================================= */
function uploadMarks() {

  if (!selectedStudent) {
    alert("Load a student first");
    return;
  }

  const semester = document.getElementById("semesterInput").value;
  const mid = document.getElementById("midInput").value;
  const end = document.getElementById("endInput").value;
  const markType = document.getElementById("markType").value;

  if (!semester) {
    alert("Enter semester");
    return;
  }

  let midValue = 0;
  let endValue = 0;

  if (markType === "both") {
    midValue = Number(mid);
    endValue = Number(end);
  }

  if (markType === "mid") {
    midValue = Number(mid);
  }

  if (markType === "end") {
    endValue = Number(end);
  }

  fetch(`${API}/faculty/upload`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      regNo: selectedStudent.regNo,
      semester,
      mid: midValue,
      end: endValue
    })
  })
  .then(res => res.json())
  .then(() => {

    alert("Marks saved successfully!");

    return fetch(`${API}/faculty/${selectedStudent.regNo}`);
  })
  .then(res => res.json())
  .then(updatedStudent => {

    selectedStudent = updatedStudent;
    selectedStudent.exams.sort((a,b)=>a.semester-b.semester);

    renderFacultyTable(selectedStudent.exams);
    renderFacultyAnalytics(selectedStudent.exams);

  })
  .catch(err => console.log(err));
}




/* ===============================
   FACULTY ANALYTICS
================================= */
function renderFacultyAnalytics(exams) {

  if (!exams.length) return;

  exams.sort((a,b)=>a.semester-b.semester);

  const labels = exams.map(e => "Sem " + e.semester);

  // overall performance formula
  const progress = exams.map(e => {
    return ((e.mid + e.end) / 2).toFixed(1);
  });

  // average calculations
  const avgMid = average(exams.map(e => e.mid));
  const avgEnd = average(exams.map(e => e.end));

  document.getElementById("facultyAvgMid").innerText = avgMid;
  document.getElementById("facultyAvgEnd").innerText = avgEnd;

  // growth condition
  const growth =
    progress[progress.length - 1] - progress[0];

  document.getElementById("growthTrend").innerText =
    growth >= 0 ? "Improving 📈" : "Declining 📉";

  if (facultyChart) facultyChart.destroy();

  facultyChart = new Chart(
    document.getElementById("facultyChart"),
    {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Overall Progress",
            data: progress,
            borderColor: "#667eea",
            backgroundColor: "rgba(102,126,234,0.2)",
            tension: 0.3,
            fill: true
          }
        ]
      },
      options: {
        responsive: true,
        plugins: {
          legend: {
            display: true
          }
        }
      }
    }
  );
}



/* ===============================
   FACULTY LOGIN
================================= */
function loginFaculty() {

  const email = document.getElementById("email").value;
  const password = document.getElementById("password").value;

  fetch(`${API}/auth/faculty-login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password })
  })
  .then(res => res.json())
  .then(data => {

    if (data.success) {
      window.location.href = "facultyDashboard.html";
    } else {
      document.getElementById("message").innerText =
        data.msg || "Invalid email or password";
    }

  })
  .catch(() => {
    document.getElementById("message").innerText =
      "Server not reachable";
  });
}
