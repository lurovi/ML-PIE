$("document").ready(function() {
    retrieveModels();
});

var divModelsContainer = $(".model-container")

divModelsContainer.click(function() {
    divModelsContainer.each(function () {
        $(this).removeClass("selected");
    });
    $(this).addClass("selected");
    let i = parseInt($("input[type=hidden]", this).val());
    let feedback = -1 + 2*i;
    provideFeedback(feedback);
    retrieveModels();
});

function retrieveModels(){
    $("#div-loading-img").attr("hidden", false)
    $.ajax({
      url: "getData",
      headers: { 'x-access-tokens': localStorage.getItem("token") }
    }).done(data => {
      formula_latex = $("h4.mb-0");
      formula_latex[0].innerHTML = data.t1;
      formula_latex[1].innerHTML = data.t2;
      updateProgressBar(data.progress);
    });
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
      console.log(data.message);
    });
}

function updateProgressBar(progressPercentage){
    progressBar = $("#evolution-progress-bar")
    progressBar.width(progressPercentage+"%")
    progressBar.attr("aria-valuenow", progressPercentage)
    progressBar.html(progressPercentage+"%")
}