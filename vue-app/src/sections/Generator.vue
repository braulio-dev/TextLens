<script setup>
import { defineEmits } from 'vue'
import { Button } from '@/components/ui/button'
import { ArrowUpTrayIcon } from '@heroicons/vue/20/solid'
import { ClipboardIcon } from '@heroicons/vue/24/outline'
import { VueSpinner } from 'vue3-spinners'
import { useRouter } from 'vue-router'
import { fileInput, isLoading, handleFileChange, triggerFileInput, handlePasteText, handlePasteImage } from '@/utils/generator'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { lang } from '@/utils/generator.js'

const handleLanguageChange = (value) => {
    lang.value = value
}

const router = useRouter()
const emit = defineEmits(['summary-received', 'block-summary'])
</script>

<template>
  <section id="generator" class="relative min-h-screen flex items-center justify-center">    
    <div class="relative z-10 flex flex-col items-center gap-10">
      <div class="bg-gradient-to-r from-blue-600 to-blue-900 text-transparent bg-clip-text">
        <h1 class="text-center text-9xl">TextLens</h1>
      </div>
      
      <div class="flex gap-7 scale-150">
        <Button class="p-3" @click="triggerFileInput" :disabled="isLoading">
          <ArrowUpTrayIcon v-if="!isLoading" />
          <VueSpinner v-else />
        </Button>
        
        <Button class="p-3" @click="() => handlePasteText(emit)" :disabled="isLoading">
          <ClipboardIcon v-if="!isLoading" />
          <VueSpinner v-else />
          <span class="text-lg" v-if="!isLoading">Pegar Texto</span>
          <span class="text-lg" v-else>Cargando...</span>
        </Button>

        <Button class="p-3" @click="() => handlePasteImage(emit)" :disabled="isLoading">
          <ClipboardIcon v-if="!isLoading" />
          <VueSpinner v-else />
          <span class="text-lg" v-if="!isLoading">Pegar Imagen</span>
          <span class="text-lg" v-else>Cargando...</span>
        </Button>
      </div>

      <div class="w-1/4">
        <Select @update:modelValue="handleLanguageChange" defaultValue="spa">
          <SelectTrigger>
            <SelectValue placeholder="Español" />
          </SelectTrigger>
          <SelectContent>
              <SelectGroup>
                <SelectItem value="spa">Español</SelectItem>
                <SelectItem value="eng">Inglés</SelectItem>
              </SelectGroup>
          </SelectContent>
        </Select>
      </div>
      
      <input type="file" ref="fileInput" class="hidden" @change="(event) => handleFileChange(event, emit)" accept=".png, .jpg, .jpeg" />
    </div>
  </section>
</template>