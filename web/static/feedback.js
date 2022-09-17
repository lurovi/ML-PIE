$("document").ready(function() {
    retrieveModels();
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
    $("#div-loading-img").attr("hidden", false)
    $.ajax({
      url: "getData",
      headers: { 'x-access-tokens': localStorage.getItem("token") }
    }).done(data => {
      if("over" in data){
        optimizationOver();
      } else {
        formula_latex = $("h4.mb-0");
        formula_latex[0].innerHTML = data.t1;
        formula_latex[1].innerHTML = data.t2;
        updateProgressBar(data.progress);
      }
    }).fail(() => {window.location = window.location.protocol + "//" + window.location.host;}
    );
    $("#div-loading-img").attr("hidden", true)
}

function provideFeedback(feedback){
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

function updateProgressBar(progressPercentage){
    progressBar = $("#evolution-progress-bar")
    progressBar.width(progressPercentage+"%")
    progressBar.attr("aria-valuenow", progressPercentage)
    progressBar.html(progressPercentage+"%")
}

function optimizationOver(){
    $("#btn-proceed").attr("disabled", false);
    $("#div-loading-img").attr("hidden", true);
    $(".model-container").attr("hidden", true);
    updateProgressBar(100);
}