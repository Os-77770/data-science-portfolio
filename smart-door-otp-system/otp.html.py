<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Smart Door – OTP Validation</title>

<style>
body {
    font-family: Arial, sans-serif;
    margin: 40px;
}
label {
    display: block;
    margin-top: 10px;
}
input {
    padding: 6px;
    width: 200px;
}
button {
    margin-top: 15px;
    padding: 8px 16px;
}
#result {
    margin-top: 20px;
    font-weight: bold;
}
</style>
</head>

<body>

<h1>Smart Door – Enter OTP</h1>

<form id="otpForm">
    <label>
        OTP Code:
        <input type="text" id="otp" required placeholder="Enter OTP">
    </label>

    <button type="submit">Validate OTP</button>
</form>

<div id="result"></div>

<script>
const API_BASE = "https://kxf1j866xf.execute-api.us-east-1.amazonaws.com/prod";

const form = document.getElementById("otpForm");
const resultDiv = document.getElementById("result");

form.addEventListener("submit", async function (e) {
    e.preventDefault();

    resultDiv.style.color = "black";
    resultDiv.textContent = "Checking OTP...";

    const payload = {
        otp: document.getElementById("otp").value.trim()
    };

    try {
        const response = await fetch(API_BASE + "/validate-otp", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const text = await response.text();
        let data;

        try {
            data = JSON.parse(text);
        } catch {
            data = { message: text };
        }

        if (data.body) {
            try {
                data = JSON.parse(data.body);
            } catch {}
        }

        const status = data.status || "error";

        resultDiv.style.color = status === "ok" ? "green" : "red";
        resultDiv.textContent = data.message || "OTP validation result";

    } catch (error) {
        resultDiv.style.color = "red";
        resultDiv.textContent = "Error: " + error.message;
    }
});
</script>

</body>
</html>