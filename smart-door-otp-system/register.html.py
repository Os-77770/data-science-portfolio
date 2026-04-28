<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Smart Door – Register Visitor</title>

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
    width: 260px;
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

<h1>Smart Door – Register Visitor</h1>

<form id="registerForm">
    <label>
        Face ID:
        <input type="text" id="faceId" required placeholder="e.g. test-face-123">
    </label>

    <label>
        Visitor Name:
        <input type="text" id="name" required placeholder="Visitor name">
    </label>

    <label>
        Phone Number (E.164):
        <input type="text" id="phoneNumber" placeholder="+13135551234">
    </label>

    <label>
        Email (optional):
        <input type="email" id="email" placeholder="visitor@example.com">
    </label>

    <button type="submit">Register Visitor</button>
</form>

<div id="result"></div>

<script>
const API_BASE = "https://kxf1j866xf.execute-api.us-east-1.amazonaws.com/prod";

const form = document.getElementById("registerForm");
const resultDiv = document.getElementById("result");

form.addEventListener("submit", async function (e) {
    e.preventDefault();

    resultDiv.style.color = "black";
    resultDiv.textContent = "Registering visitor and sending OTP...";

    const payload = {
        faceId: document.getElementById("faceId").value.trim(),
        name: document.getElementById("name").value.trim(),
        phoneNumber: document.getElementById("phoneNumber").value.trim() || null,
        email: document.getElementById("email").value.trim() || null
    };

    try {
        const response = await fetch(API_BASE + "/register-visitor", {
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

        resultDiv.style.color = response.ok ? "green" : "red";
        resultDiv.textContent = data.message || "Visitor registered successfully";

    } catch (error) {
        resultDiv.style.color = "red";
        resultDiv.textContent = "Error: " + error.message;
    }
});
</script>

</body>
</html>