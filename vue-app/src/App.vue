<template>
  <div class="relative min-h-screen">
    <div class="fixed inset-0 z-0">
      <div class="bg-gradient-to-r from-purple-400 to-blue-300 rounded-full w-2/4 h-[15%] fixed -right-[40%] top-[25%]"></div>
      <div class="bg-gradient-to-r from-blue-500 to-purple-600 rounded-full w-2/4 h-[25%] fixed -left-[25%] top-[25%]"></div>
      <div class="bg-gradient-to-r from-purple-600 to-blue-500 rounded-full w-2/4 h-[17%] fixed -left-[40%] top-[58%]"></div>
      <div class="bg-gradient-to-r from-blue-500 to-purple-600 rounded-full w-2/4 h-[27%] fixed -right-[27%] top-[65%]"></div>
    </div>
    
    <div class="relative z-10">
      <router-view></router-view>
    </div>
  </div>
</template>

<style>
body {
  margin: 0;
  padding: 0;
  overflow-x: hidden;
}
</style>