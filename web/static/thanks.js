$("document").ready(function() {
    if (localStorage.getItem("username") === null) {
        window.location = window.location.protocol + "//" + window.location.host + "/";
    }
});