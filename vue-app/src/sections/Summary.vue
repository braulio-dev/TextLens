<script setup>
import { defineProps } from 'vue'

const props = defineProps({
  extracted_text: String,
  summary: String
})
</script>

<template>
  <section id="summary">
    <div class="flex flex-col items-center gap-10">
      <div class="bg-gradient-to-r from-purple-500 to-purple-900 text-transparent bg-clip-text">
        <h1 class="text-center text-9xl">Resultados</h1>
      </div>
            
      <div class="p-5 bg-card shadow-2xl rounded-xl w-1/2">
        <span class="text-xl bg-gradient-to-r from-blue-500 to-blue-900 text-transparent bg-clip-text">
          OCR (Entrenado)
        </span>

        <div class="shadow-md mt-1 m-5 bg-gradient-to-b from-white to-slate-100 rounded-2xl">
          <div class="p-4">
            <span class="text-card-foreground text-lg text-justify">
              {{ extracted_text }}
            </span>
          </div>
        </div>

        <span class="text-xl bg-gradient-to-r from-blue-500 to-blue-900 text-transparent bg-clip-text">
          Resumen (Pre-entrenado)
        </span>
        <p class="mt-2 text-card-foreground text-xl text-justify">{{ summary }}</p>
      </div>
      <input type="file" ref="fileInput" class="hidden" @change="handleFileChange" accept=".png, .jpg, .jpeg" />
    </div>
  </section>
</template>
