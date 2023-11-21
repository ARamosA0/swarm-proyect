var app = angular.module('catsvsdogs', []);
var socket = io.connect();


app.controller('statsCtrl', function($scope){
  $scope.id = 0;
  $scope.name = 0;
  $scope.value = 0;

  var updateScores = function(){
    socket.on('recomendaciones', function (json) {
      var data = JSON.parse(json);
      console.log(data)
      $scope.$apply(function () {
         $scope.id = data.id;
         $scope.name = data.name;
         $scope.value = data.value;
       });
    });
  };

  var init = function(){
    document.body.style.opacity=1;
    updateScores();
  };
  socket.on('message',function(data){
    init();
  });
});

