const availableProblems = ["boston", "windspeed"];
const availableProblemNames = ["Boston Housing", "Wind Speed"];

$("document").ready(function() {
    if(localStorage.getItem("token") === null){
        window.location = window.location.protocol + "//" + window.location.host + "/";
    }
    if(localStorage.getItem("next") != null){
        $('#put-submit-btn-text-here').text("Proceed with " + availableProblemNames[availableProblems.indexOf(localStorage.getItem("next"))]);
    }
    retrieveSurveyModels();
});

$("#btn-undo").on("click", function(){
    $.ajax({
      url: "reset",
      headers: { 'x-access-tokens': localStorage.getItem("token") }
    });
    $.ajax({
      url: "startRun/" + localStorage.getItem("problem"),
      headers: { 'x-access-tokens': localStorage.getItem("token") }
    }).done( data => {
          localStorage.setItem("token", data.id);
          localStorage.getItem("token")
          window.location = window.location.protocol + "//" + window.location.host + "/feedback";
        }
    ).fail(() => {window.location = window.location.protocol + "//" + window.location.host;}
    );
})

$("#btn-submit").on("click", function(){
    submitSurvey();
    if(localStorage.getItem("next") != null){
        localStorage.setItem("problem", localStorage.getItem("next"));
        localStorage.removeItem("next");
        localStorage.setItem("oldToken", localStorage.getItem("token"));
        $.ajax({
            url: "startRun/" + localStorage.getItem("problem")
        }).done( data => {
          localStorage.setItem("token", data.id);
          localStorage.setItem("over", false);
          localStorage.setItem("problem", data.problem);
          window.location = window.location.protocol + "//" + window.location.host + "/feedback";
        }
    );
    } else {
        window.location = window.location.protocol + "//" + window.location.host + "/thanks";
    }
})

function retrieveSurveyModels(){
    $.ajax({
      url: "getSurveyData",
      headers: { 'x-access-tokens': localStorage.getItem("token") }
    }).done(data => {
        let finalDivContent = ''
        data.comparisons.forEach((model, i) => {
          let first_model = "$$" + model.pie_latex + "$$";
          let second_model = "$$" + model.other_latex + "$$";
          let first = "pie";
          let second = model.type;
          let label = model.type;
          let divContentTemplate = `<div class=div-models-container-template> \
                              <div class="row mb-2"> \
                                  <div class="col-md-6 sbox model-container"> \
                                      <div \
                                              class="row g-0 border rounded overflow-hidden flex-md-row mb-4 shadow-sm h-md-250 position-relative"> \
                                          <div class="col p-4 d-flex flex-column position-static"> \
                                              <strong class="d-inline-block mb-2 text-primary">Model 1</strong> \
                                              <h4 class="mb-0 model0_online">${first_model}</h4> \
                                              <input type="hidden" value="0"> \
                                          </div> \
                                          <div class="col-auto d-none d-lg-block"> \
                                              <h2 class="interpretability">1</h2> \
                                          </div> \
                                      </div> \
                                  </div> \
                                  <div class="col-md-6 sbox  model-container"> \
                                      <div \
                                              class="row g-0 border rounded overflow-hidden flex-md-row mb-4 shadow-sm h-md-250 position-relative"> \
                                          <div class="col p-4 d-flex flex-column position-static"> \
                                              <strong class="d-inline-block mb-2 text-primary">Model 2</strong> \
                                              <h4 class="mb-0 model0_size">${second_model}</h4> \
                                              <input type="hidden" value="1"> \
                                          </div> \
                                          <div class="col-auto d-none d-lg-block"> \
                                              <h2 class="interpretability">2</h2> \
                                          </div> \
                                      </div> \
                                  </div> \
           \
           \
                                  <div class="btn-group" role="group0" aria-label="Basic radio toggle button group"> \
                                      <input type="radio" class="btn-check" name="${label}" id="${label}_${first}" \
                                             autocomplete="off"> \
                                      <label class="btn btn-outline-primary" for="${label}_${first}">The left one is more \
                                          interpretable</label> \
           \
                                      <input type="radio" class="btn-check" name="${label}" id="${label}_same" \
                                             autocomplete="off" checked> \
                                      <label class="btn btn-outline-primary" for="${label}_same">They are equally \
                                          (un)interpretable</label> \
           \
                                      <input type="radio" class="btn-check" name="${label}" id="${label}_${second}" \
                                             autocomplete="off"> \
                                      <label class="btn btn-outline-primary" for="${label}_${second}">The right one is more \
                                          interpretable</label> \
                                  </div> \
                              </div> \
                              </div>`;
          finalDivContent = finalDivContent + divContentTemplate;
        });
        $("#div-survey-container").html(finalDivContent);
        MathJax.typeset();
        $("#div-loading-img").attr("hidden", true);
    });
}

function submitSurvey(){
    let survey = {}
    $(":radio").each(function () {
        if ($(this).is(':checked')) survey[this.name] = this.id.substring(this.id.indexOf('_') + 1);
    });
    $.ajax({
      type: "POST",
      url: "answerSurvey",
      headers: { 'x-access-tokens': localStorage.getItem("token") },
      data: JSON.stringify(survey),
      contentType: "application/json"
    }).fail(() => {window.location = window.location.protocol + "//" + window.location.host;}
    );
}
