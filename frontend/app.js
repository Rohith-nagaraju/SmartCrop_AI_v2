/* =========================================================
   SMARTCROP AI
   Complete Frontend JavaScript
   Authentication + Live Weather + AI Analysis
   Risk Decomposition + Recommendations + History
   ========================================================= */

"use strict";

/* =========================================================
   GLOBAL STATE
   ========================================================= */

const $ = (id) => document.getElementById(id);

let currentFarmer = null;
let authToken = null;
let weatherData = null;
let latestAnalysis = null;

const FARMER_STORAGE_KEY = "smartcrop_farmer";
const TOKEN_STORAGE_KEY = "smartcrop_token";


/* =========================================================
   STORAGE
   ========================================================= */

function saveFarmer(farmer) {
    try {
        localStorage.setItem(
            FARMER_STORAGE_KEY,
            JSON.stringify(farmer)
        );
    } catch (error) {
        console.error("Could not save farmer:", error);
    }
}


function loadFarmer() {
    try {
        const value =
            localStorage.getItem(FARMER_STORAGE_KEY);

        return value
            ? JSON.parse(value)
            : null;

    } catch (error) {
        console.error("Could not load farmer:", error);
        return null;
    }
}


function clearFarmer() {
    localStorage.removeItem(FARMER_STORAGE_KEY);
    localStorage.removeItem(TOKEN_STORAGE_KEY);

    currentFarmer = null;
    authToken = null;
}


function saveToken(token) {
    authToken = token;

    if (token) {
        localStorage.setItem(
            TOKEN_STORAGE_KEY,
            token
        );
    }
}


function loadToken() {
    try {
        return localStorage.getItem(
            TOKEN_STORAGE_KEY
        );
    } catch (error) {
        return null;
    }
}


/* =========================================================
   API HELPER
   ========================================================= */

async function apiFetch(url, options = {}) {

    const headers =
        new Headers(options.headers || {});

    if (
        authToken &&
        !headers.has("Authorization")
    ) {
        headers.set(
            "Authorization",
            `Bearer ${authToken}`
        );
    }

    const isFormData =
        options.body instanceof FormData;

    if (
        !isFormData &&
        options.body &&
        !headers.has("Content-Type")
    ) {
        headers.set(
            "Content-Type",
            "application/json"
        );
    }

    const response =
        await fetch(url, {
            ...options,
            headers
        });

    let data = null;

    try {
        data = await response.json();
    } catch (error) {
        data = null;
    }

    if (!response.ok) {

        const message =
            data?.detail ||
            data?.message ||
            `Request failed (${response.status})`;

        throw new Error(message);
    }

    return data;
}


/* =========================================================
   AUTH SCREEN
   ========================================================= */

function showAuthScreen() {

    $("authScreen")?.classList.remove("hidden");
    $("appScreen")?.classList.add("hidden");
}


function showAppScreen() {

    $("authScreen")?.classList.add("hidden");
    $("appScreen")?.classList.remove("hidden");
}


/* =========================================================
   AUTH MESSAGES
   ========================================================= */

function showAuthMessage(
    elementId,
    message,
    type = "error"
) {

    const element = $(elementId);

    if (!element) {
        return;
    }

    element.textContent = message;

    element.classList.remove(
        "error",
        "success",
        "hidden"
    );

    element.classList.add(type);
}


function clearAuthMessage(elementId) {

    const element = $(elementId);

    if (!element) {
        return;
    }

    element.textContent = "";

    element.classList.add("hidden");

    element.classList.remove(
        "error",
        "success"
    );
}


/* =========================================================
   LOGIN / REGISTER SWITCH
   ========================================================= */

function showLoginForm() {

    $("loginFormContainer")
        ?.classList.remove("hidden");

    $("registerFormContainer")
        ?.classList.add("hidden");

    $("showLoginButton")
        ?.classList.add("active");

    $("showRegisterButton")
        ?.classList.remove("active");

    document
        .querySelectorAll('[data-auth-tab="login"]')
        .forEach((button) => {
            button.classList.add("active");
            button.setAttribute(
                "aria-selected",
                "true"
            );
        });

    document
        .querySelectorAll('[data-auth-tab="register"]')
        .forEach((button) => {
            button.classList.remove("active");
            button.setAttribute(
                "aria-selected",
                "false"
            );
        });

    clearAuthMessage("loginMessage");
    clearAuthMessage("registerMessage");
}


function showRegisterForm() {

    $("loginFormContainer")
        ?.classList.add("hidden");

    $("registerFormContainer")
        ?.classList.remove("hidden");

    $("showLoginButton")
        ?.classList.remove("active");

    $("showRegisterButton")
        ?.classList.add("active");

    document
        .querySelectorAll('[data-auth-tab="login"]')
        .forEach((button) => {
            button.classList.remove("active");
            button.setAttribute(
                "aria-selected",
                "false"
            );
        });

    document
        .querySelectorAll('[data-auth-tab="register"]')
        .forEach((button) => {
            button.classList.add("active");
            button.setAttribute(
                "aria-selected",
                "true"
            );
        });

    clearAuthMessage("loginMessage");
    clearAuthMessage("registerMessage");
}


/* =========================================================
   REGISTER FARMER
   ========================================================= */

async function registerFarmer(event) {

    event?.preventDefault();

    clearAuthMessage("registerMessage");

    const fullName =
        $("registerName")?.value.trim();

    const email =
        $("registerEmail")?.value.trim();

    const password =
        $("registerPassword")?.value;

    const farmName =
        $("registerFarmName")?.value.trim();

    const location =
        $("registerLocation")?.value.trim();

    const crop =
        $("registerCrop")?.value;

    const fieldId =
        $("registerFieldId")?.value.trim();


    if (!fullName) {
        showAuthMessage(
            "registerMessage",
            "Please enter your full name."
        );
        return;
    }

    if (!email) {
        showAuthMessage(
            "registerMessage",
            "Please enter your email."
        );
        return;
    }

    if (!password || password.length < 6) {
        showAuthMessage(
            "registerMessage",
            "Password must contain at least 6 characters."
        );
        return;
    }

    if (!farmName) {
        showAuthMessage(
            "registerMessage",
            "Please enter your farm name."
        );
        return;
    }

    if (!location) {
        showAuthMessage(
            "registerMessage",
            "Please enter your farm location."
        );
        return;
    }

    if (!crop) {
        showAuthMessage(
            "registerMessage",
            "Please select your crop."
        );
        return;
    }

    if (!fieldId) {
        showAuthMessage(
            "registerMessage",
            "Please enter a field ID."
        );
        return;
    }


    const button =
        document.querySelector(
            "#registerForm button[type='submit']"
        );

    const originalText =
        button?.textContent ||
        "Create Account";

    if (button) {
        button.disabled = true;
        button.textContent =
            "Creating account...";
    }


    try {

        const data =
            await apiFetch(
                "/api/auth/register",
                {
                    method: "POST",

                    body: JSON.stringify({
                        full_name: fullName,
                        email: email,
                        password: password,
                        farm_name: farmName,
                        location: location,
                        crop: crop,
                        field_id: fieldId
                    })
                }
            );


        if (data.token) {
            saveToken(data.token);
        }


        currentFarmer =
            data.farmer || {
                full_name: fullName,
                email: email,
                farm_name: farmName,
                location: location,
                crop: crop,
                field_id: fieldId
            };


        saveFarmer(currentFarmer);


        showAuthMessage(
            "registerMessage",
            "Account created successfully.",
            "success"
        );


        setTimeout(() => {
            initializeDashboard();
        }, 400);


    } catch (error) {

        console.error(
            "Registration error:",
            error
        );

        showAuthMessage(
            "registerMessage",
            error.message ||
            "Registration failed."
        );

    } finally {

        if (button) {
            button.disabled = false;
            button.textContent =
                originalText;
        }
    }
}


/* =========================================================
   LOGIN
   ========================================================= */

async function loginFarmer(event) {

    event?.preventDefault();

    clearAuthMessage("loginMessage");

    const email =
        $("loginEmail")?.value.trim();

    const password =
        $("loginPassword")?.value;


    if (!email) {
        showAuthMessage(
            "loginMessage",
            "Please enter your email."
        );
        return;
    }

    if (!password) {
        showAuthMessage(
            "loginMessage",
            "Please enter your password."
        );
        return;
    }


    const button =
        document.querySelector(
            "#loginForm button[type='submit']"
        );

    const originalText =
        button?.textContent ||
        "Login";


    if (button) {
        button.disabled = true;
        button.textContent =
            "Logging in...";
    }


    try {

        const data =
            await apiFetch(
                "/api/auth/login",
                {
                    method: "POST",

                    body: JSON.stringify({
                        email,
                        password
                    })
                }
            );


        if (!data.token) {
            throw new Error(
                "Login succeeded but no session token was returned."
            );
        }


        saveToken(data.token);

        currentFarmer =
            data.farmer || {
                email
            };

        saveFarmer(currentFarmer);

        await initializeDashboard();


    } catch (error) {

        console.error(
            "Login error:",
            error
        );

        clearFarmer();

        showAuthMessage(
            "loginMessage",
            error.message ||
            "Login failed."
        );

    } finally {

        if (button) {
            button.disabled = false;
            button.textContent =
                originalText;
        }
    }
}


/* =========================================================
   GET CURRENT FARMER
   ========================================================= */

async function getCurrentFarmer() {

    if (!authToken) {
        return null;
    }

    try {

        const data =
            await apiFetch(
                "/api/auth/me"
            );

        return (
            data?.farmer ||
            data ||
            null
        );

    } catch (error) {

        console.error(
            "Could not retrieve farmer:",
            error
        );

        return null;
    }
}


/* =========================================================
   LOGOUT
   ========================================================= */

async function logoutFarmer() {

    try {

        if (authToken) {

            await apiFetch(
                "/api/auth/logout",
                {
                    method: "POST"
                }
            );
        }

    } catch (error) {

        console.warn(
            "Logout request failed:",
            error
        );

    } finally {

        clearFarmer();

        weatherData = null;
        latestAnalysis = null;

        showAuthScreen();
        showLoginForm();

        $("loginForm")?.reset();

        $("result")
            ?.classList.add("hidden");
    }
}


/* =========================================================
   DASHBOARD
   ========================================================= */

async function initializeDashboard() {

    showAppScreen();

    const remoteFarmer =
        await getCurrentFarmer();


    if (remoteFarmer) {

        currentFarmer =
            remoteFarmer;

        saveFarmer(
            currentFarmer
        );
    }


    if (!currentFarmer) {

        showAuthScreen();
        showLoginForm();

        return;
    }


    updateDashboardHeader();
    updateLocationPanel();

    initializeAnalysis();

    await loadWeather();
    await loadAnalysisHistory();
}


/* =========================================================
   HEADER
   ========================================================= */

function updateDashboardHeader() {

    if (!currentFarmer) {
        return;
    }

    const name =
        currentFarmer.full_name ||
        currentFarmer.name ||
        "Farmer";

    const farm =
        currentFarmer.farm_name ||
        "My Farm";


    if ($("headerFarmerName")) {
        $("headerFarmerName")
            .textContent = name;
    }

    if ($("headerFarmName")) {
        $("headerFarmName")
            .textContent = farm;
    }
}


/* =========================================================
   LOCATION
   ========================================================= */

function updateLocationPanel() {

    if (!currentFarmer) {
        return;
    }

    const location =
        currentFarmer.location_name ||
        currentFarmer.location ||
        [
            currentFarmer.city,
            currentFarmer.state,
            currentFarmer.country
        ]
            .filter(Boolean)
            .join(", ");


    if ($("farmLocation")) {

        $("farmLocation")
            .textContent =
            location ||
            "Farm location not available";
    }
}


/* =========================================================
   WEATHER
   ========================================================= */

async function loadWeather() {

    const status =
        $("weatherStatus");

    const forecast =
        $("weatherForecast");


    if (status) {
        status.textContent =
            "Loading live weather...";
    }

    if (forecast) {
        forecast.innerHTML = "";
    }


    try {

        const data =
            await apiFetch(
                "/api/weather"
            );

        weatherData =
            data;

        renderWeather(data);


    } catch (error) {

        console.error(
            "Weather error:",
            error
        );

        if (status) {
            status.textContent =
                "Weather could not be loaded.";
        }

        if (forecast) {

            forecast.innerHTML = `
                <div class="weather-error">
                    <strong>Weather unavailable</strong>
                    <p>
                        ${escapeHtml(
                            error.message ||
                            "Please try again later."
                        )}
                    </p>
                </div>
            `;
        }
    }
}


/* =========================================================
   WEATHER RENDER
   ========================================================= */

function renderWeather(data) {

    const container =
        $("weatherForecast");

    if (!container) {
        return;
    }


    let days =
        data?.daily ||
        data?.forecast ||
        data?.days ||
        data?.daily_weather ||
        [];


    if (!Array.isArray(days)) {

        if (Array.isArray(days?.forecast)) {
            days = days.forecast;
        } else {
            days =
                Object.values(days || {});
        }
    }


    if (!days.length) {

        container.innerHTML = `
            <div class="weather-error">
                Weather data is unavailable.
            </div>
        `;

        return;
    }


    container.innerHTML =
        days
            .slice(0, 5)
            .map(
                (day, index) =>
                    createWeatherCard(
                        day,
                        index
                    )
            )
            .join("");


    if ($("weatherStatus")) {

        $("weatherStatus")
            .textContent =
            "✓ Weather updated from your saved farm location";
    }


    updateWeatherUpdateTime(data);
}


/* =========================================================
   WEATHER CARD
   ========================================================= */

function createWeatherCard(day, index) {

    const label =
        index === 0
            ? "Today"
            : index === 1
                ? "Tomorrow"
                : `Day ${index + 1}`;


    const date =
        firstValue(
            day,
            [
                "date",
                "datetime",
                "day"
            ]
        );


    const temperature =
        firstValue(
            day,
            [
                "temperature",
                "temperature_max",
                "temp",
                "max_temperature",
                "temp_max",
                "temperature_2m_max"
            ]
        );


    const humidity =
        firstValue(
            day,
            [
                "humidity",
                "relative_humidity",
                "relative_humidity_mean",
                "humidity_mean"
            ]
        );


    const rainfall =
        firstValue(
            day,
            [
                "rainfall",
                "precipitation",
                "precipitation_sum",
                "rain",
                "rain_mm"
            ]
        );


    const weatherCode =
        firstValue(
            day,
            [
                "weather_code",
                "weathercode",
                "weatherCode",
                "code"
            ]
        );


    return `
        <div class="weather-card">

            <div class="weather-day">
                ${escapeHtml(label)}
            </div>

            ${
                date
                    ? `
                        <div class="weather-date">
                            ${formatWeatherDate(date)}
                        </div>
                      `
                    : ""
            }

            <div class="weather-icon">
                ${weatherIcon(
                    weatherCode,
                    rainfall
                )}
            </div>

            <div class="weather-temperature">
                ${escapeHtml(
                    formatNumber(
                        temperature,
                        "°C"
                    )
                )}
            </div>

            <div class="weather-details">

                <div>
                    💧
                    ${escapeHtml(
                        formatNumber(
                            humidity,
                            "%"
                        )
                    )}
                    RH
                </div>

                <div>
                    🌧️
                    ${escapeHtml(
                        formatNumber(
                            rainfall,
                            " mm"
                        )
                    )}
                </div>

            </div>

        </div>
    `;
}


/* =========================================================
   WEATHER HELPERS
   ========================================================= */

function firstValue(object, keys) {

    for (const key of keys) {

        if (
            object &&
            object[key] !== undefined &&
            object[key] !== null
        ) {
            return object[key];
        }
    }

    return null;
}


function formatNumber(
    value,
    suffix = ""
) {

    if (
        value === null ||
        value === undefined ||
        value === ""
    ) {
        return "--";
    }

    const number =
        Number(value);

    if (!Number.isFinite(number)) {
        return "--";
    }

    return (
        Math.round(number * 10) / 10
    ) + suffix;
}


function weatherIcon(
    code,
    rainfall
) {

    const numericCode =
        Number(code);


    if (Number.isFinite(numericCode)) {

        if (numericCode === 0) {
            return "☀️";
        }

        if (numericCode <= 3) {
            return "🌤️";
        }

        if (
            numericCode >= 51 &&
            numericCode <= 67
        ) {
            return "🌧️";
        }

        if (
            numericCode >= 80 &&
            numericCode <= 82
        ) {
            return "🌦️";
        }

        if (numericCode >= 95) {
            return "⛈️";
        }
    }


    if (Number(rainfall) > 5) {
        return "🌧️";
    }

    if (Number(rainfall) > 0) {
        return "🌦️";
    }

    return "🌤️";
}


function formatWeatherDate(value) {

    if (!value) {
        return "";
    }

    const date =
        new Date(value);

    if (
        Number.isNaN(
            date.getTime()
        )
    ) {
        return String(value);
    }

    return date.toLocaleDateString(
        undefined,
        {
            day: "numeric",
            month: "short"
        }
    );
}


function updateWeatherUpdateTime(data) {

    const element =
        $("weatherUpdateTime");

    if (!element) {
        return;
    }


    const updated =
        data?.updated_at ||
        data?.updatedAt ||
        data?.generated_at ||
        data?.timestamp;


    if (updated) {

        const date =
            new Date(updated);

        if (
            !Number.isNaN(
                date.getTime()
            )
        ) {

            element.textContent =
                `Last weather update: ${
                    date.toLocaleTimeString(
                        undefined,
                        {
                            hour: "numeric",
                            minute: "2-digit"
                        }
                    )
                }`;

            return;
        }
    }


    element.textContent =
        "Weather updated just now";
}


/* =========================================================
   ANALYSIS INITIALIZATION
   ========================================================= */

function initializeAnalysis() {

    initializeAnalysisForm();
    initializeImagePreview();
    initializeCropSelection();
}


function initializeAnalysisForm() {

    const form =
        $("form");

    if (!form) {
        return;
    }

    if (
        form.dataset.initialized === "true"
    ) {
        return;
    }

    form.dataset.initialized =
        "true";

    form.addEventListener(
        "submit",
        submitAnalysis
    );
}


/* =========================================================
   IMAGE PREVIEW
   ========================================================= */

function initializeImagePreview() {

    const input =
        $("image");

    if (!input) {
        return;
    }

    if (
        input.dataset.initialized === "true"
    ) {
        return;
    }

    input.dataset.initialized =
        "true";


    input.addEventListener(
        "change",
        () => {

            const file =
                input.files?.[0];

            const filename =
                $("filename");

            const preview =
                $("inputPreview");


            if (!file) {

                if (filename) {
                    filename.textContent =
                        "No image selected";
                }

                if (preview) {
                    preview.classList.add(
                        "hidden"
                    );

                    preview.removeAttribute(
                        "src"
                    );
                }

                return;
            }


            if (filename) {
                filename.textContent =
                    file.name;
            }


            if (preview) {

                const url =
                    URL.createObjectURL(
                        file
                    );

                preview.src =
                    url;

                preview.classList.remove(
                    "hidden"
                );

                preview.onload =
                    () => {
                        URL.revokeObjectURL(
                            url
                        );
                    };
            }
        }
    );
}


/* =========================================================
   CROP SELECTION
   ========================================================= */

function initializeCropSelection() {

    const crop =
        $("crop");

    const registeredCrop =
        currentFarmer?.crop;


    if (
        crop &&
        registeredCrop
    ) {

        const option =
            Array.from(
                crop.options
            ).find(
                (item) =>
                    item.value.toLowerCase() ===
                    registeredCrop.toLowerCase()
            );


        if (option) {
            crop.value =
                option.value;
        }
    }
}


/* =========================================================
   SUBMIT ANALYSIS
   ========================================================= */

async function submitAnalysis(event) {

    event.preventDefault();


    const imageInput =
        $("image");

    const cropInput =
        $("crop");

    const fieldInput =
        $("field");

    const stageInput =
        $("stage");


    if (!imageInput?.files?.length) {

        alert(
            "Please select a crop leaf image."
        );

        return;
    }


    if (!cropInput?.value) {

        alert(
            "Please select a crop."
        );

        return;
    }


    if (!authToken) {

        alert(
            "Your session has expired. Please login again."
        );

        showAuthScreen();
        showLoginForm();

        return;
    }


    const button =
        $("analyzeButton");

    const buttonText =
        $("buttonText");

    const originalText =
        buttonText?.textContent ||
        "Run AI Analysis";


    if (button) {
        button.disabled = true;
    }

    if (buttonText) {
        buttonText.textContent =
            "Analyzing...";
    }


    const formData =
        new FormData();


    formData.append(
        "image",
        imageInput.files[0]
    );


    formData.append(
        "crop",
        cropInput.value
    );


    formData.append(
        "field_id",
        fieldInput?.value ||
        currentFarmer?.field_id ||
        "FIELD-001"
    );


    formData.append(
        "growth_stage",
        stageInput?.value ||
        "Vegetative"
    );


    /*
       IMPORTANT:
       Weather values are NOT sent here.

       The backend gets:
       temperature
       humidity
       rainfall
       soil moisture

       from the farmer's saved location.
    */


    try {

        const data =
            await apiFetch(
                "/api/analyze",
                {
                    method: "POST",
                    body: formData
                }
            );


        latestAnalysis =
            data;


        if (data?.crop_mismatch) {
            showMismatchResult(data);
        } else {
            showMainResult(data);
        }


    } catch (error) {

        console.error(
            "Analysis error:",
            error
        );


        if (
            error.message
                .toLowerCase()
                .includes("login") ||
            error.message
                .toLowerCase()
                .includes("session") ||
            error.message
                .toLowerCase()
                .includes("unauthorized")
        ) {

            clearFarmer();

            showAuthScreen();
            showLoginForm();

            return;
        }


        alert(
            error.message ||
            "Analysis failed. Please try again."
        );


    } finally {

        if (button) {
            button.disabled = false;
        }

        if (buttonText) {
            buttonText.textContent =
                originalText;
        }
    }
}


/* =========================================================
   SHOW MAIN RESULT
   ========================================================= */

function showMainResult(data) {

    $("result")
        ?.classList.remove("hidden");


    /* -----------------------------
       DISEASE
       ----------------------------- */

    const disease =
        data?.disease ||
        data?.prediction ||
        data?.predicted_class ||
        "Unknown";


    if ($("disease")) {

        $("disease")
            .textContent =
            data?.uncertain
                ? "UNCERTAIN"
                : disease;
    }


    if ($("disease-status")) {

        if (data?.uncertain) {

            $("disease-status")
                .textContent =
                "Prediction requires verification";

        } else if (data?.model_version) {

            $("disease-status")
                .textContent =
                `AI classification • Model ${data.model_version}`;

        } else {

            $("disease-status")
                .textContent =
                "AI crop-health classification";
        }
    }


    /* -----------------------------
       CONFIDENCE
       ----------------------------- */

    const confidence =
        Number(
            data?.confidence ??
            data?.confidence_score ??
            0
        );


    if ($("conf")) {

        $("conf")
            .textContent =
            `${formatPercent(confidence)}%`;
    }


    if ($("confidenceBar")) {

        const percentage =
            confidence <= 1
                ? confidence * 100
                : confidence;

        $("confidenceBar")
            .style.width =
            `${clamp(
                percentage,
                0,
                100
            )}%`;
    }


    /* -----------------------------
       SEVERITY
       ----------------------------- */

    const severityPercent =
        data?.severity_percent;

    const severityClass =
        data?.severity_class;


    if ($("severity")) {

        if (
            severityPercent !== null &&
            severityPercent !== undefined &&
            Number.isFinite(
                Number(severityPercent)
            )
        ) {

            $("severity")
                .textContent =
                `${Number(
                    severityPercent
                ).toFixed(2)}% ${
                    severityClass || ""
                }`;

        } else {

            $("severity")
                .textContent =
                "Not assessed";
        }
    }


    /* -----------------------------
       CURRENT RISK
       ----------------------------- */

    const risk =
        Number(
            data?.risk_score ??
            data?.risk ??
            data?.current_risk ??
            0
        );


    if ($("risk")) {

        $("risk")
            .textContent =
            `${formatPercent(risk)}%`;
    }


    if ($("risk-status")) {

        $("risk-status")
            .textContent =
            data?.risk_level &&
            data.risk_level !== "NOT_ASSESSED"
                ? data.risk_level
                : getRiskDescription(risk);
    }


    $("mismatch")
        ?.classList.add("hidden");


    /* -----------------------------
       RISK DECOMPOSITION
       ----------------------------- */

    renderRiskDecomposition(data);


    /* -----------------------------
       EXPLANATION
       ----------------------------- */

    renderExplanation(data);


    /* -----------------------------
       FORECAST
       ----------------------------- */

    renderForecast(data);


    /* -----------------------------
       RECOMMENDATION
       ----------------------------- */

    renderRecommendation(data);


    /* -----------------------------
       IMAGE QUALITY
       ----------------------------- */

    renderImageQuality(data);


    /* -----------------------------
       GRAD-CAM
       ----------------------------- */

    setGradcam(data);


    latestAnalysis =
        data;


    loadAnalysisHistory();


    scrollToResults();
}


/* =========================================================
   MISMATCH RESULT
   ========================================================= */

function showMismatchResult(data) {

    $("result")
        ?.classList.remove("hidden");


    if ($("mismatch")) {

        $("mismatch")
            .innerHTML = `
                <strong>
                    ⚠️ Crop / Image mismatch
                </strong>

                <p>
                    The uploaded image could not be
                    reliably matched to the selected crop.
                    The risk assessment has therefore
                    been withheld.
                </p>
            `;

        $("mismatch")
            .classList.remove("hidden");
    }


    if ($("disease")) {
        $("disease")
            .textContent =
            "Image could not be reliably matched";
    }


    if ($("disease-status")) {
        $("disease-status")
            .textContent =
            "Classification not reliable";
    }


    if ($("conf")) {
        $("conf")
            .textContent =
            "Not reliable";
    }


    if ($("confidenceBar")) {
        $("confidenceBar")
            .style.width =
            "0%";
    }


    if ($("severity")) {
        $("severity")
            .textContent =
            "Not assessed";
    }


    if ($("risk")) {
        $("risk")
            .textContent =
            "Not assessed";
    }


    if ($("risk-status")) {
        $("risk-status")
            .textContent =
            "Risk withheld";
    }


    renderRiskDecomposition({
        risk_decomposition: null
    });


    if ($("explain")) {

        $("explain")
            .innerHTML = `
                <div class="explanation-item">
                    <strong>
                        Result withheld
                    </strong>

                    <p>
                        The model did not produce a
                        sufficiently reliable crop-image
                        match, so downstream disease-risk
                        interpretation should not be treated
                        as valid.
                    </p>
                </div>
            `;
    }


    if ($("forecast")) {

        $("forecast")
            .innerHTML = `
                <div class="forecast-empty">
                    Five-day projection withheld because
                    the crop/image match was not reliable.
                </div>
            `;
    }


    renderRecommendation(data);
    renderImageQuality(data);
    setGradcam(data);

    scrollToResults();
}


/* =========================================================
   RISK DESCRIPTION
   ========================================================= */

function getRiskDescription(risk) {

    const value =
        Number(risk);


    if (!Number.isFinite(value)) {
        return "Risk not available";
    }


    if (value < 20) {
        return "LOW";
    }

    if (value < 40) {
        return "MODERATE";
    }

    if (value < 60) {
        return "ELEVATED";
    }

    if (value < 80) {
        return "HIGH";
    }

    return "CRITICAL";
}


/* =========================================================
   RISK DECOMPOSITION
   ========================================================= */

function renderRiskDecomposition(data) {

    const container =
        $("riskDecomposition");

    if (!container) {
        return;
    }


    const decomposition =
        data?.risk_decomposition ||
        data?.risk_components ||
        data?.decomposition;


    let values = [];


    /* -----------------------------
       BACKEND DECOMPOSITION
       ----------------------------- */

    if (
        decomposition &&
        typeof decomposition === "object" &&
        !Array.isArray(decomposition)
    ) {

        values =
            Object.entries(
                decomposition
            );
    }


    /* -----------------------------
       FALLBACK
       ----------------------------- */

    if (!values.length) {

        const confidence =
            Number(
                data?.confidence ??
                data?.confidence_score ??
                0
            );


        const severity =
            Number(
                data?.severity_percent
            );


        const dpi =
            Number(
                data?.dpi
            );


        if (Number.isFinite(confidence)) {

            const confidencePercent =
                confidence <= 1
                    ? confidence * 100
                    : confidence;


            values.push([
                "Disease",
                0.40 *
                confidencePercent
            ]);
        }


        if (Number.isFinite(severity)) {

            const normalizedSeverity =
                clamp(
                    severity / 25,
                    0,
                    1
                ) * 100;


            values.push([
                "Severity",
                0.30 *
                normalizedSeverity
            ]);
        }


        if (Number.isFinite(dpi)) {

            values.push([
                "Environment",
                0.30 * dpi
            ]);
        }
    }


    if (!values.length) {

        container.innerHTML = `
            <div class="decomposition-empty">
                Risk-factor breakdown is not available
                for this analysis.
            </div>
        `;

        return;
    }


    const labels = {

        disease:
            "Disease",

        severity:
            "Severity",

        temperature:
            "Temperature",

        humidity:
            "Humidity",

        rainfall:
            "Rainfall",

        precipitation:
            "Rainfall",

        soil_moisture:
            "Soil Moisture",

        environment:
            "Environment"
    };


    container.innerHTML =
        values
            .slice(0, 6)
            .map(
                ([key, value]) => {

                    const normalizedKey =
                        String(key)
                            .toLowerCase()
                            .replace(
                                /\s+/g,
                                "_"
                            );


                    const label =
                        labels[
                            normalizedKey
                        ] ||
                        prettifyLabel(
                            key
                        );


                    const number =
                        Number(value);


                    const display =
                        Number.isFinite(number)
                            ? formatSigned(number)
                            : String(value);


                    return `
                        <div class="risk-factor">

                            <span>
                                ${escapeHtml(
                                    label
                                )}
                            </span>

                            <strong>
                                ${escapeHtml(
                                    display
                                )}
                            </strong>

                        </div>
                    `;
                }
            )
            .join("");
}


/* =========================================================
   RECOMMENDED ACTION
   ========================================================= */

function renderRecommendation(data) {

    const element =
        $("rec");

    if (!element) {
        return;
    }


    const plan =
        data?.recommendation_plan ||
        data?.recommendationPlan ||
        {};


    const disease =
        plan?.disease ||
        data?.disease ||
        "Crop-health condition";


    const crop =
        plan?.crop ||
        currentFarmer?.crop ||
        "";


    const priority =
        plan?.priority ||
        "MONITOR CLOSELY";


    const summary =
        plan?.summary ||
        data?.recommendation ||
        "Continue field monitoring and verify the assessment with field observations.";


    const immediateActions =
        Array.isArray(
            plan?.immediate_actions
        )
            ? plan.immediate_actions
            : [];


    const organicManagement =
        Array.isArray(
            plan?.organic_management
        )
            ? plan.organic_management
            : [];


    const avoid =
        Array.isArray(
            plan?.avoid
        )
            ? plan.avoid
            : [];


    const weatherContext =
        plan?.weather_context ||
        {};


    const weatherMessage =
        weatherContext?.message ||
        "Use forecast conditions as supporting information and confirm field conditions before acting.";


    const growthStageNote =
        plan?.growth_stage_note;


    const organicDisclaimer =
        plan?.organic_disclaimer ||
        "Organic-management options are decision-support guidance, not a product prescription. Use only locally approved inputs and follow applicable product-label and certification requirements.";


    const renderList =
        (items, emptyText) => {

            if (
                !Array.isArray(items) ||
                !items.length
            ) {

                return `
                    <p class="recommendation-empty">
                        ${escapeHtml(
                            emptyText
                        )}
                    </p>
                `;
            }


            return `
                <ul class="recommendation-list">

                    ${
                        items
                            .map(
                                (item) => `
                                    <li class="recommendation-list-item">
                                        ${escapeHtml(
                                            String(item)
                                        )}
                                    </li>
                                `
                            )
                            .join("")
                    }

                </ul>
            `;
        };


    const weatherDetails = [];


    if (
        weatherContext?.humidity !== undefined &&
        weatherContext?.humidity !== null
    ) {

        weatherDetails.push(
            `Humidity ${formatNumber(
                weatherContext.humidity,
                "%"
            )}`
        );
    }


    if (
        weatherContext?.rainfall !== undefined &&
        weatherContext?.rainfall !== null
    ) {

        weatherDetails.push(
            `Rainfall ${formatNumber(
                weatherContext.rainfall,
                " mm"
            )}`
        );
    }


    if (
        weatherContext?.soil_moisture !== undefined &&
        weatherContext?.soil_moisture !== null
    ) {

        weatherDetails.push(
            `Soil-moisture proxy ${formatNumber(
                weatherContext.soil_moisture,
                "%"
            )}`
        );
    }


    if (
        weatherContext?.temperature !== undefined &&
        weatherContext?.temperature !== null
    ) {

        weatherDetails.push(
            `Temperature ${formatNumber(
                weatherContext.temperature,
                "°C"
            )}`
        );
    }


    element.innerHTML = `

        <div class="recommendation-panel">

            <div class="recommendation-priority">
                ${escapeHtml(
                    String(priority)
                )}
            </div>


            <div class="recommendation-summary">

                <strong>
                    ${escapeHtml(
                        String(disease)
                    )}
                </strong>


                ${
                    crop
                        ? `
                            <span class="recommendation-context">
                                ${escapeHtml(
                                    String(crop)
                                )}
                            </span>
                          `
                        : ""
                }


                ${
                    data?.risk_level
                        ? `
                            <span class="recommendation-context">
                                Risk:
                                ${escapeHtml(
                                    String(
                                        data.risk_level
                                    )
                                )}
                            </span>
                          `
                        : ""
                }


                <p>
                    ${escapeHtml(
                        String(summary)
                    )}
                </p>

            </div>


            <div class="recommendation-grid">

                <section class="recommendation-section">

                    <h4>
                        ⚡ Immediate Actions
                    </h4>

                    ${renderList(
                        immediateActions,
                        "Increase field scouting and verify affected plants."
                    )}

                </section>


                <section class="recommendation-section organic-section">

                    <h4>
                        🌱 Organic &amp; Integrated Management
                    </h4>

                    ${renderList(
                        organicManagement,
                        "Maintain sanitation, canopy airflow and regular field scouting."
                    )}

                </section>


                <section class="recommendation-section avoid-section">

                    <h4>
                        ⚠️ Avoid
                    </h4>

                    ${renderList(
                        avoid,
                        "Avoid acting on the image result alone when field observations disagree."
                    )}

                </section>

            </div>


            <div class="recommendation-weather">

                <strong>
                    🌦 Weather Consideration
                </strong>


                ${
                    weatherDetails.length
                        ? `
                            <div class="recommendation-weather-values">

                                ${
                                    weatherDetails
                                        .map(
                                            (item) => `
                                                <span>
                                                    ${escapeHtml(
                                                        item
                                                    )}
                                                </span>
                                            `
                                        )
                                        .join("")
                                }

                            </div>
                          `
                        : ""
                }


                <p>
                    ${escapeHtml(
                        String(
                            weatherMessage
                        )
                    )}
                </p>

            </div>


            ${
                growthStageNote
                    ? `
                        <div class="recommendation-stage">

                            <strong>
                                🌿 Growth-stage guidance
                            </strong>

                            <p>
                                ${escapeHtml(
                                    String(
                                        growthStageNote
                                    )
                                )}
                            </p>

                        </div>
                      `
                    : ""
            }


            <div class="recommendation-disclaimer">

                <strong>
                    📌 Decision-support note
                </strong>

                <p>
                    ${escapeHtml(
                        String(
                            organicDisclaimer
                        )
                    )}
                </p>

            </div>

        </div>
    `;
}


/* =========================================================
   EXPLANATION
   ========================================================= */

function renderExplanation(data) {

    const container =
        $("explain");

    if (!container) {
        return;
    }


    const explanation =
        data?.explanation ||
        {};


    const environment =
        explanation?.environment ||
        data?.environment ||
        data?.current_weather ||
        {};


    const temperature =
        environment?.temperature;

    const humidity =
        environment?.humidity;

    const rainfall =
        environment?.rainfall;

    const soil =
        environment?.soil_moisture;


    const growthStage =
        data?.growth_stage ||
        explanation?.growth_stage ||
        "Not specified";


    const severityNote =
        explanation?.severity_note ||
        data?.severity_note;


    const forecastNote =
        explanation?.forecast_note ||
        data?.forecast_note;


    let html = `
        <div class="explanation-grid">
    `;


    html += explanationItem(
        "🎯",
        "Disease confidence",
        `${formatPercent(
            data?.confidence || 0
        )}%`
    );


    if (
        data?.severity_percent !== null &&
        data?.severity_percent !== undefined
    ) {

        html += explanationItem(
            "🧬",
            "Prototype severity",
            `${Number(
                data.severity_percent
            ).toFixed(2)}% ${
                data?.severity_class || ""
            }`
        );
    }


    if (
        data?.dpi !== null &&
        data?.dpi !== undefined
    ) {

        html += explanationItem(
            "🌦️",
            "Environmental pressure",
            Number(data.dpi).toFixed(2)
        );
    }


    if (
        temperature !== null &&
        temperature !== undefined
    ) {

        html += explanationItem(
            "🌡️",
            "Temperature",
            formatNumber(
                temperature,
                " °C"
            )
        );
    }


    if (
        humidity !== null &&
        humidity !== undefined
    ) {

        html += explanationItem(
            "💧",
            "Humidity",
            formatNumber(
                humidity,
                " %"
            )
        );
    }


    if (
        rainfall !== null &&
        rainfall !== undefined
    ) {

        html += explanationItem(
            "🌧️",
            "Rainfall",
            formatNumber(
                rainfall,
                " mm"
            )
        );
    }


    if (
        soil !== null &&
        soil !== undefined
    ) {

        html += explanationItem(
            "🌱",
            "Soil moisture proxy",
            formatNumber(
                soil,
                " %"
            )
        );
    }


    html += explanationItem(
        "🌱",
        "Growth stage",
        growthStage
    );


    html += `
        </div>
    `;


    if (severityNote) {

        html += `
            <div class="result-note">

                <strong>
                    Severity:
                </strong>

                ${escapeHtml(
                    severityNote
                )}

            </div>
        `;
    }


    if (forecastNote) {

        html += `
            <div class="result-note">

                <strong>
                    Projection:
                </strong>

                ${escapeHtml(
                    forecastNote
                )}

            </div>
        `;
    }


    container.innerHTML =
        html;
}


function explanationItem(
    icon,
    label,
    value
) {

    return `
        <div class="explanation-item">

            <div class="explanation-icon">
                ${icon}
            </div>

            <div>

                <span>
                    ${escapeHtml(
                        label
                    )}
                </span>

                <strong>
                    ${escapeHtml(
                        String(value)
                    )}
                </strong>

            </div>

        </div>
    `;
}


/* =========================================================
   FIVE-DAY RISK FORECAST
   ========================================================= */

function renderForecast(data) {

    const container =
        $("forecast");

    if (!container) {
        return;
    }


    let days =
        data?.forecast ||
        data?.risk_forecast ||
        data?.projection ||
        data?.five_day_projection ||
        [];


    if (
        !Array.isArray(days) &&
        typeof days === "object"
    ) {

        days =
            days.days ||
            days.forecast ||
            days.projection ||
            [];
    }


    if (!Array.isArray(days)) {
        days = [];
    }


    if (!days.length) {

        container.innerHTML = `
            <div class="forecast-empty">
                Five-day risk projection is not available.
            </div>
        `;

        return;
    }


    container.innerHTML =
        days
            .slice(0, 5)
            .map(
                (day, index) => {

                    const risk =
                        Number(
                            day?.risk ??
                            day?.risk_score ??
                            day?.projected_risk ??
                            day?.value ??
                            0
                        );


                    const label =
                        day?.label ||
                        day?.day ||
                        `Day ${index + 1}`;


                    const temperature =
                        firstValue(
                            day,
                            [
                                "temperature",
                                "temperature_max",
                                "temp",
                                "temperature_2m_max"
                            ]
                        );


                    const humidity =
                        firstValue(
                            day,
                            [
                                "humidity",
                                "relative_humidity"
                            ]
                        );


                    const rainfall =
                        firstValue(
                            day,
                            [
                                "rainfall",
                                "precipitation",
                                "precipitation_sum",
                                "rain_mm"
                            ]
                        );


                    return `
                        <div class="forecast-card">

                            <div class="forecast-day">
                                ${escapeHtml(
                                    label
                                )}
                            </div>

                            <div class="forecast-risk">
                                ${formatPercent(
                                    risk
                                )}
                            </div>

                            <div class="forecast-risk-label">
                                ${getRiskDescription(
                                    risk
                                )}
                            </div>


                            ${
                                temperature !== null &&
                                temperature !== undefined
                                    ? `
                                        <div class="forecast-weather">
                                            🌡️
                                            ${formatNumber(
                                                temperature,
                                                "°C"
                                            )}
                                        </div>
                                      `
                                    : ""
                            }


                            ${
                                humidity !== null &&
                                humidity !== undefined
                                    ? `
                                        <div class="forecast-weather">
                                            💧
                                            ${formatNumber(
                                                humidity,
                                                "%"
                                            )}
                                        </div>
                                      `
                                    : ""
                            }


                            ${
                                rainfall !== null &&
                                rainfall !== undefined
                                    ? `
                                        <div class="forecast-weather">
                                            🌧️
                                            ${formatNumber(
                                                rainfall,
                                                " mm"
                                            )}
                                        </div>
                                      `
                                    : ""
                            }

                        </div>
                    `;
                }
            )
            .join("");
}


/* =========================================================
   IMAGE QUALITY
   ========================================================= */

function renderImageQuality(data) {

    const container =
        $("quality");

    if (!container) {
        return;
    }


    const quality =
        data?.image_quality ||
        data?.quality;


    if (!quality) {

        container.innerHTML = `
            <div class="quality-empty">
                Image quality information is not available.
            </div>
        `;

        return;
    }


    if (
        typeof quality === "string"
    ) {

        container.innerHTML = `
            <div class="quality-result">
                ${escapeHtml(
                    quality
                )}
            </div>
        `;

        return;
    }


    const score =
        quality.score ??
        quality.quality_score ??
        quality.value;


    const label =
        quality.label ||
        quality.status ||
        formatQuality(score);


    let html = `
        <div class="quality-result">

            <div class="quality-main">

                <strong>
                    ${escapeHtml(
                        String(label)
                    )}
                </strong>

                ${
                    score !== undefined
                        ? `
                            <span>
                                Score:
                                ${formatPercent(
                                    Number(score)
                                )}
                            </span>
                          `
                        : ""
                }

            </div>
    `;


    const details =
        quality.details ||
        quality.reasons ||
        quality.messages;


    if (Array.isArray(details)) {

        html += `
            <ul class="quality-details">

                ${
                    details
                        .map(
                            (item) =>
                                `<li>${escapeHtml(
                                    String(item)
                                )}</li>`
                        )
                        .join("")
                }

            </ul>
        `;
    }


    html += `
        </div>
    `;


    container.innerHTML =
        html;
}


function formatQuality(value) {

    const score =
        Number(value);


    if (!Number.isFinite(score)) {
        return "Image quality assessed";
    }


    if (score >= 80) {
        return "Good";
    }

    if (score >= 60) {
        return "Acceptable";
    }

    if (score >= 40) {
        return "Fair";
    }

    return "Poor";
}


/* =========================================================
   GRAD-CAM
   ========================================================= */

function setGradcam(data) {

    const image =
        $("preview");

    if (!image) {
        return;
    }


    const gradcam =
        data?.gradcam ||
        data?.grad_cam ||
        data?.gradcam_image ||
        data?.explanation_image;


    if (!gradcam) {

        image.removeAttribute("src");

        image.classList.add(
            "hidden"
        );

        return;
    }


    let source =
        String(gradcam);


    if (
        !source.startsWith("data:")
    ) {

        source =
            `data:image/png;base64,${source}`;
    }


    image.src =
        source;

    image.classList.remove(
        "hidden"
    );
}


/* =========================================================
   HISTORY
   ========================================================= */

async function loadAnalysisHistory() {

    const container =
        $("analysisHistory");

    if (!container) {
        return;
    }


    try {

        const data =
            await apiFetch(
                "/api/history"
            );


        let history =
            data?.history ||
            data?.analyses ||
            data ||
            [];


        if (!Array.isArray(history)) {
            history = [];
        }


        if (!history.length) {

            container.innerHTML = `
                <div class="history-empty">
                    No previous analyses yet.
                </div>
            `;

            return;
        }


        container.innerHTML =
            history
                .slice(0, 10)
                .map(
                    createHistoryItem
                )
                .join("");


    } catch (error) {

        console.error(
            "History error:",
            error
        );


        container.innerHTML = `
            <div class="history-empty">
                Analysis history could not be loaded.
            </div>
        `;
    }
}


function createHistoryItem(item) {

    const disease =
        item?.disease ||
        item?.prediction ||
        "Unknown";


    const crop =
        item?.crop ||
        "Unknown crop";


    const risk =
        item?.risk ??
        item?.risk_score;


    const confidence =
        item?.confidence ??
        item?.confidence_score;


    const severity =
        item?.severity ||
        item?.severity_class ||
        "—";


    const timestamp =
        item?.created_at ||
        item?.timestamp ||
        item?.date;


    return `
        <div class="history-item">

            <div class="history-main">

                <strong>
                    ${escapeHtml(
                        String(disease)
                    )}
                </strong>

                <span>
                    ${escapeHtml(
                        String(crop)
                    )}
                </span>

            </div>


            <div class="history-values">

                ${
                    risk !== undefined &&
                    risk !== null
                        ? `
                            <span>
                                Risk:
                                <strong>
                                    ${formatPercent(
                                        Number(risk)
                                    )}%
                                </strong>
                            </span>
                          `
                        : ""
                }


                ${
                    confidence !== undefined &&
                    confidence !== null
                        ? `
                            <span>
                                Confidence:
                                <strong>
                                    ${formatPercent(
                                        Number(
                                            confidence
                                        )
                                    )}%
                                </strong>
                            </span>
                          `
                        : ""
                }


                <span>
                    Severity:
                    <strong>
                        ${escapeHtml(
                            String(severity)
                        )}
                    </strong>
                </span>

            </div>


            ${
                timestamp
                    ? `
                        <div class="history-date">
                            ${formatDateTime(
                                timestamp
                            )}
                        </div>
                      `
                    : ""
            }

        </div>
    `;
}


/* =========================================================
   HELPERS
   ========================================================= */

function scrollToResults() {

    const result =
        $("result");

    if (!result) {
        return;
    }


    setTimeout(() => {

        result.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });

    }, 100);
}


function escapeHtml(value) {

    return String(
        value ?? ""
    )
        .replace(
            /&/g,
            "&amp;"
        )
        .replace(
            /</g,
            "&lt;"
        )
        .replace(
            />/g,
            "&gt;"
        )
        .replace(
            /"/g,
            "&quot;"
        )
        .replace(
            /'/g,
            "&#039;"
        );
}


function prettifyLabel(value) {

    return String(value)
        .replace(
            /_/g,
            " "
        )
        .replace(
            /\b\w/g,
            (letter) =>
                letter.toUpperCase()
        );
}


function formatPercent(value) {

    let number =
        Number(value);


    if (!Number.isFinite(number)) {
        return "0";
    }


    /*
       Confidence normally arrives as 0–1.
       Risk normally arrives as 0–100.
    */

    if (
        number >= 0 &&
        number <= 1
    ) {
        number *= 100;
    }


    number =
        clamp(
            number,
            0,
            100
        );


    return (
        Math.round(
            number * 10
        ) / 10
    ).toString();
}


function formatSigned(value) {

    const number =
        Number(value);


    if (!Number.isFinite(number)) {
        return String(value);
    }


    const rounded =
        Math.round(
            number * 10
        ) / 10;


    if (rounded > 0) {
        return `+${rounded}`;
    }


    return String(
        rounded
    );
}


function clamp(
    value,
    min,
    max
) {

    return Math.min(
        Math.max(
            value,
            min
        ),
        max
    );
}


function formatDateTime(value) {

    const date =
        new Date(value);


    if (
        Number.isNaN(
            date.getTime()
        )
    ) {
        return String(value);
    }


    return date.toLocaleString(
        undefined,
        {
            day: "numeric",
            month: "short",
            year: "numeric",
            hour: "numeric",
            minute: "2-digit"
        }
    );
}


/* =========================================================
   EVENT BINDINGS
   ========================================================= */

function bindEvents() {

    const loginForm =
        $("loginForm");

    if (loginForm) {

        loginForm.addEventListener(
            "submit",
            loginFarmer
        );
    }


    const registerForm =
        $("registerForm");

    if (registerForm) {

        registerForm.addEventListener(
            "submit",
            registerFarmer
        );
    }


    const showRegister =
        $("showRegisterButton");

    if (showRegister) {

        showRegister.addEventListener(
            "click",
            (event) => {

                event.preventDefault();

                showRegisterForm();
            }
        );
    }


    const showLogin =
        $("showLoginButton");

    if (showLogin) {

        showLogin.addEventListener(
            "click",
            (event) => {

                event.preventDefault();

                showLoginForm();
            }
        );
    }


    document
        .querySelectorAll(
            '[data-auth-tab="login"]'
        )
        .forEach(
            (button) => {

                button.addEventListener(
                    "click",
                    showLoginForm
                );
            }
        );


    document
        .querySelectorAll(
            '[data-auth-tab="register"]'
        )
        .forEach(
            (button) => {

                button.addEventListener(
                    "click",
                    showRegisterForm
                );
            }
        );


    const logout =
        $("logoutButton");

    if (logout) {

        logout.addEventListener(
            "click",
            logoutFarmer
        );
    }
}


/* =========================================================
   START APPLICATION
   ========================================================= */

async function startApplication() {

    bindEvents();


    const storedToken =
        loadToken();

    const storedFarmer =
        loadFarmer();


    if (
        storedToken &&
        storedFarmer
    ) {

        authToken =
            storedToken;

        currentFarmer =
            storedFarmer;


        const remoteFarmer =
            await getCurrentFarmer();


        if (remoteFarmer) {

            currentFarmer =
                remoteFarmer;

            saveFarmer(
                currentFarmer
            );

            await initializeDashboard();

            return;
        }


        clearFarmer();
    }


    showAuthScreen();
    showLoginForm();
}


/* =========================================================
   DOM READY
   ========================================================= */

if (
    document.readyState ===
    "loading"
) {

    document.addEventListener(
        "DOMContentLoaded",
        startApplication
    );

} else {

    startApplication();
}