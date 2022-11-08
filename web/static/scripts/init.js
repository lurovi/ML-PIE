const availableProblems = ["boston", "heating"];
const availableProblemNames = ["Boston Housing", "Heating Efficiency"];

$("document").ready(function() {
    let chosenProblemIndex = Math.floor(Math.random() * availableProblems.length);
    let chosenProblem = availableProblems[chosenProblemIndex];
    let nextProblem = availableProblems[availableProblems.length - 1 - chosenProblemIndex];
    localStorage.setItem("problem", chosenProblem);
    localStorage.setItem("next", nextProblem);
    $('#put-problem-name-here').text(availableProblemNames[chosenProblemIndex]);
});

$("#btn-proceed").on("click",function() {
    $.ajax({
      url: "startRun/" + localStorage.getItem("problem")
    }).done( data => {
          localStorage.setItem("token", data.id);
          localStorage.setItem("over", false);
          localStorage.setItem("problem", data.problem);
          window.location = window.location.protocol + "//" + window.location.host + "/feedback";
        }
    ).fail(() => {$("#btn-proceed").attr("disabled", false);}
    );
    $("#btn-proceed").attr("disabled", true);
})