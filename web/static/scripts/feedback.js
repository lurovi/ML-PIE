const progressRetrievalInterval = setInterval(function(){
    retrieveProgress();
}, 3000);

let lastProgressPercentage = 0;
let lastModels = ["",""];
let lastFeedback = 1;
let retryCounter = 0;
const maxRetries = 1;

$(window).resize(function () {
    w = $("h4.mb-0").width()/4;
});

$("document").ready(function() {
    if(localStorage.getItem("over") === "false"){
      retrieveModels();
    }
    lastProgressPercentage = 0;
});

$(".model-container").click(function() {
    $(".model-container").each(function () {
        $(this).removeClass("selected");
    });
    $(this).addClass("selected");
    let i = parseInt($("input[type=hidden]", this).val());
    let feedback = -1 + 2*i;
    provideFeedback(feedback);
    retrieveModels();
});

$("#btn-proceed").on("click", function(){
    window.location = window.location.protocol + "//" + window.location.host + "/survey";
})

function retrieveModels(){
    $("#div-models-container").attr("hidden", true);
    $("#div-loading-img").attr("hidden", false);
    $.ajax({
      url: "getData",
      headers: { 'x-access-tokens': localStorage.getItem("token") }
    }).done(data => {
      if("wait" in data){
        setTimeout(function() { retrieveModels(); }, 1000);
        return;
      }
      if("over" in data){
        optimizationOver();
      } else {
        // check for equality first
        let equals = true;
        for (var i = 0; i < 2; i++) {
            if (!(lastModels[i] === data.models[i]["latex"])){
                equals = false;
            }
        }
        if (equals && retryCounter < maxRetries){
            console.log("Equal models encountered, trying to fetch again.");
            retryCounter = retryCounter + 1;
            return retrieveModels();
        }
        else if (equals && retryCounter >= maxRetries){
            console.log("Equal models encountered, sending back the same feedback.");
            provideFeedback(lastFeedback);
            return retrieveModels();
        }
        // models
        $("#div-models-container").attr("hidden", false);
        $("#div-loading-img").attr("hidden", true);
        formula_latex = $("h4.mb-0");
        for (var i = 0; i < 2; i++) {
          let new_model = data.models[i]["latex"];
          lastModels[i] = data.models[i]["latex"]
          formula_latex[i].innerHTML = "$$" + new_model + "$$";
          w = $("h4.mb-0").width()/5;
          f_size = Math.min(1.5, w / new_model.length);
          $(formula_latex[i]).attr("style", "font-size: " + f_size + "em;");
        }
        MathJax.typeset();
        updateProgressBar(data.progress);
      }
    }).fail(() => {window.location = window.location.protocol + "//" + window.location.host;}
    );
}

function provideFeedback(feedback){
    lastFeedback = feedback;
    $.ajax({
      type: "POST",
      url: "provideFeedback",
      headers: { 'x-access-tokens': localStorage.getItem("token") },
      data: JSON.stringify({ 'feedback': feedback }),
      contentType: "application/json"
    }).done(data => {
      if("over" in data){
        optimizationOver();
      }
    }).fail(() => {window.location = window.location.protocol + "//" + window.location.host;}
    );
}

function retrieveProgress(){
    if(localStorage.getItem("over") === "true"){
      return;
    }
    $.ajax({
      url: "getProgress",
      headers: { 'x-access-tokens': localStorage.getItem("token") }
    }).done(data => {
      updateProgressBar(data.progress);
      if(data.progress >= 100){
        optimizationOver();
      }
    }).fail(() => {window.location = window.location.protocol + "//" + window.location.host;}
    );
}

function updateProgressBar(progressPercentage){
    lastProgressPercentage = Math.max(lastProgressPercentage, progressPercentage);
    progressBar = $("#evolution-progress-bar");
    progressBar.width(lastProgressPercentage+"%");
    progressBar.attr("aria-valuenow", lastProgressPercentage);
    progressBar.html(lastProgressPercentage+"%");
}

function optimizationOver(){
    localStorage.setItem("over", true);
    clearInterval(progressRetrievalInterval);
    $("#btn-proceed").attr("disabled", false);
    $("#div-loading-img").attr("hidden", true);
    $(".model-container").attr("hidden", true);
    updateProgressBar(100);
}
